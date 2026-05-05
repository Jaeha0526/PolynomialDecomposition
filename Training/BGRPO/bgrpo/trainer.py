"""
BGRPO trainer — the main policy-update loop.

The flow matches the paper's §3.5 description exactly:

    for each batch of ``num_questions`` prompts:
        1. Beam-roll out each prompt (beam width = ``num_generations``),
           producing per-token log-probs under π_θold.
        2. Score each beam completion for correctness; compute reward +
           advantage per prompt.
        3. For ``num_iterations`` PPO policy updates:
             a. For each (prompt, beam) pair, re-forward the concatenated
                prompt+completion sequence under the current policy π_θ
                (with gradients) and the frozen reference π_ref.
             b. Extract per-token log-probs at the completion positions.
             c. Compute PPO-clipped + KL loss via :class:`GRPOObjective`,
                accumulate across the batch, backward, optimizer step.

Gradient accumulation is explicit (we average loss across the
``num_questions`` prompts before each optimizer step). This keeps the
per-update gradient norm independent of the batch size.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import wandb

from ._utils import forward_logits
from .objective import GRPOObjective, ObjectiveConfig, gather_token_logprobs
from .reward import RewardConfig, build_reward, compute_advantages
from .rollout import BeamRollout, RolloutResult, SamplingRollout

logger = logging.getLogger(__name__)

CorrectnessFn = Callable[[str, str], bool]  # (prompt_str, completion_str) -> is_correct


@dataclass
class BGRPOConfig:
    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    # Rollout
    num_generations: int = 32         # beam_width (BGRPO) or num_samples (GRPO)
    max_new_tokens: int = 150
    rollout_temperature: float = 1.0
    top_k: Optional[int] = None
    use_beam: bool = True             # True = BGRPO (beam), False = GRPO (sampling)
    # Training schedule
    num_questions: int = 8            # prompts per batch
    num_iterations: int = 5           # policy updates per batch of prompts
    total_training_samples: int = 200  # paper §3.5: "200 non-repeating problems"
    gradient_norm_clip: Optional[float] = 1.0
    # Reward + objective
    reward: RewardConfig = field(default_factory=RewardConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    # Checkpointing
    save_steps: int = 5               # save every N outer iterations (used when save_at_steps is None)
    # Explicit absolute-step list. When set, overrides save_steps: only save
    # if labelled_outer is in this list. Lets runs save non-uniform ckpts
    # (e.g. dense early, sparse late) without writing intermediates.
    save_at_steps: Optional[list[int]] = None
    output_dir: Optional[Path] = None
    # For continuation runs: label the first outer iteration as
    # ``start_outer_step + 1`` so ckpt names / wandb step continue from a
    # prior run. The prompt-pool iterator and the PPO loop itself still
    # start from scratch — this is purely a labelling offset.
    start_outer_step: int = 0
    # wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_config: dict = field(default_factory=dict)
    # Reproducibility
    seed: int = 148


class BGRPOTrainer:
    """Single-GPU BGRPO trainer. Model + tokenizer are passed in ready."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        prompts: Iterable[str],
        correctness_fn: CorrectnessFn,
        config: BGRPOConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = list(prompts)
        self.correctness_fn = correctness_fn
        self.cfg = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model.to(self.device)

        # Frozen reference policy = deepcopy of the starting model.
        self.ref_model = copy.deepcopy(model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if config.use_beam:
            self.rollout = BeamRollout(
                tokenizer=tokenizer,
                beam_width=config.num_generations,
                max_new_tokens=config.max_new_tokens,
                temperature=config.rollout_temperature,
                top_k=config.top_k,
            )
        else:
            # GRPO mode: samples are iid, so the ordering the rollout returns
            # carries no "rank" meaning. Refusing rank-style rewards here
            # surfaces the conceptual mismatch instead of silently training
            # on a decay applied to an arbitrary sample order.
            if config.reward.kind in ("rank", "reverse_rank"):
                raise ValueError(
                    f"reward.kind={config.reward.kind!r} requires use_beam=True; "
                    "sampled rollouts have no meaningful rank ordering."
                )
            self.rollout = SamplingRollout(
                tokenizer=tokenizer,
                num_samples=config.num_generations,
                max_new_tokens=config.max_new_tokens,
                temperature=config.rollout_temperature,
                top_k=config.top_k,
            )

        # Reward config's beam_width must match the rollout's num_generations.
        if config.reward.beam_width != config.num_generations:
            raise ValueError(
                f"RewardConfig.beam_width ({config.reward.beam_width}) must equal "
                f"BGRPOConfig.num_generations ({config.num_generations})."
            )
        self.reward_fn = build_reward(config.reward, device=self.device)
        self.objective = GRPOObjective(config.objective)

        self.global_step = 0
        self.outer_iter = 0

    # --- main loop -----------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        random.seed(cfg.seed)
        if cfg.wandb_project:
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                       config={**cfg.wandb_config,
                               "num_generations": cfg.num_generations,
                               "num_questions": cfg.num_questions,
                               "num_iterations": cfg.num_iterations,
                               "learning_rate": cfg.learning_rate,
                               "kl_beta": cfg.objective.kl_beta,
                               "clip_epsilon": cfg.objective.clip_epsilon,
                               "reward_kind": cfg.reward.kind})

        batch_iter = self._iter_prompt_batches()
        for outer, batch in enumerate(batch_iter):
            # `labelled_outer` is 1-indexed and offset by start_outer_step so
            # continuation runs produce ckpt-205/-210/... instead of -5/-10.
            labelled_outer = cfg.start_outer_step + outer + 1
            self.outer_iter = labelled_outer
            self._run_one_outer_iteration(batch)
            if cfg.output_dir:
                if cfg.save_at_steps is not None:
                    should_save = labelled_outer in cfg.save_at_steps
                else:
                    should_save = (outer + 1) % cfg.save_steps == 0
                if should_save:
                    self._save_checkpoint(f"checkpoint-{labelled_outer}")

        if cfg.output_dir:
            self._save_checkpoint("checkpoint-final")
        if cfg.wandb_project:
            wandb.finish()

    # --- per-outer-iteration pieces ------------------------------------------

    def _iter_prompt_batches(self) -> Iterable[list[str]]:
        """Yield batches of ``num_questions`` prompts, looping as needed.

        Paper §3.5 says "200 non-repeating problems". We emit ceil(total /
        num_questions) batches, sampling without replacement from the prompt
        pool. If the pool is smaller than ``total_training_samples``, we
        recycle in shuffled passes.
        """
        rng = random.Random(self.cfg.seed)
        pool = list(self.prompts)
        rng.shuffle(pool)
        seen = 0
        cursor = 0
        while seen < self.cfg.total_training_samples:
            if cursor + self.cfg.num_questions > len(pool):
                rng.shuffle(pool)
                cursor = 0
            batch = pool[cursor : cursor + self.cfg.num_questions]
            cursor += self.cfg.num_questions
            seen += self.cfg.num_questions
            yield batch

    def _run_one_outer_iteration(self, prompts: list[str]) -> None:
        """Roll out all prompts, then do ``num_iterations`` policy updates."""
        # Phase 1: rollouts + π_ref scoring — no gradients, current θ = θ_old.
        #
        # Caching π_ref log-probs here saves num_iterations-1 full ref forwards
        # per prompt: π_ref is frozen, so re-running it inside the inner loop
        # for every policy update is pure waste.
        rollouts: list[RolloutResult] = []
        ref_logprobs_per_prompt: list[torch.Tensor] = []
        advantages_per_prompt: list[torch.Tensor] = []
        rewards_per_prompt: list[torch.Tensor] = []
        correct_per_prompt: list[float] = []
        completion_lengths: list[int] = []
        correctness_summary = {"n_correct": 0, "n_total": 0}

        for prompt_str in prompts:
            prompt_ids = self._encode_prompt(prompt_str).to(self.device)
            ro = self.rollout(self.model, prompt_ids)

            correct_mask = torch.tensor(
                [self.correctness_fn(prompt_str, c) for c in ro.beam_strings],
                device=self.device,
                dtype=torch.float32,
            )
            rewards = self.reward_fn(correct_mask)
            advantages = compute_advantages(rewards)

            ref_logprobs = self._score_reference(ro)

            rollouts.append(ro)
            ref_logprobs_per_prompt.append(ref_logprobs)
            advantages_per_prompt.append(advantages)
            rewards_per_prompt.append(rewards.detach())
            correct_per_prompt.append(float(correct_mask.mean().item()))
            # Per-rollout effective completion length (tokens up to first PAD).
            if ro.completion_mask.numel() > 0:
                completion_lengths.extend(
                    int(x) for x in ro.completion_mask.sum(dim=1).detach().cpu().tolist()
                )
            correctness_summary["n_correct"] += int(correct_mask.sum().item())
            correctness_summary["n_total"] += int(correct_mask.numel())

        # Rollout-level stats (constant across the num_iterations inner updates
        # of this outer step — rollouts are frozen during Phase 2).
        all_rewards = torch.cat(rewards_per_prompt) if rewards_per_prompt else torch.zeros(0)
        import numpy as _np
        lengths_arr = _np.array(completion_lengths, dtype=_np.float32) if completion_lengths else _np.zeros(0)
        rollout_stats = {
            "bgrpo/reward_mean": float(all_rewards.mean().item()) if all_rewards.numel() else 0.0,
            "bgrpo/reward_std": float(all_rewards.std().item()) if all_rewards.numel() > 1 else 0.0,
            "bgrpo/reward_min": float(all_rewards.min().item()) if all_rewards.numel() else 0.0,
            "bgrpo/reward_max": float(all_rewards.max().item()) if all_rewards.numel() else 0.0,
            "bgrpo/length_mean": float(lengths_arr.mean()) if lengths_arr.size else 0.0,
            "bgrpo/length_std": float(lengths_arr.std()) if lengths_arr.size > 1 else 0.0,
            "bgrpo/length_max": float(lengths_arr.max()) if lengths_arr.size else 0.0,
            "bgrpo/per_prompt_correct_mean": float(_np.mean(correct_per_prompt)) if correct_per_prompt else 0.0,
            "bgrpo/per_prompt_correct_nonzero": float(sum(1 for x in correct_per_prompt if x > 0)),
        }

        # Phase 2: `num_iterations` PPO updates on the fixed rollouts.
        for inner in range(self.cfg.num_iterations):
            self.optimizer.zero_grad(set_to_none=True)
            acc_tele: dict[str, float] = {}
            for ro, adv, ref_lp in zip(rollouts, advantages_per_prompt, ref_logprobs_per_prompt):
                loss, tele = self._compute_loss(ro, adv, ref_lp)
                (loss / len(rollouts)).backward()
                for k, v in tele.items():
                    acc_tele[k] = acc_tele.get(k, 0.0) + v / len(rollouts)

            # Measure the raw gradient norm BEFORE clipping so wandb shows the
            # actual magnitude the clipper saw — useful for diagnosing when
            # clipping is biting vs. idle.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.gradient_norm_clip or float("inf"),
            )
            acc_tele["bgrpo/grad_norm"] = float(grad_norm.item())
            self.optimizer.step()

            payload = {**acc_tele, **rollout_stats}
            self._log_step(payload, correctness_summary, inner)
            self.global_step += 1

    # --- compute helpers -----------------------------------------------------

    @torch.no_grad()
    def _score_reference(self, ro: RolloutResult) -> torch.Tensor:
        """Run π_ref once on the rollout's (prompt + completion) sequences."""
        w, comp_len = ro.completion_ids.shape
        if comp_len == 0:
            return torch.zeros(w, 0, device=self.device)
        full_ids, prompt_len = self._build_full_ids(ro)
        ref_logits, _ = forward_logits(self.ref_model, full_ids)
        ref_completion_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :]
        return gather_token_logprobs(ref_completion_logits, ro.completion_ids)

    def _compute_loss(self, ro: RolloutResult, advantages: torch.Tensor,
                      ref_logprobs: torch.Tensor):
        """Re-score completion tokens under π_θ (grad) and combine with cached π_ref."""
        w, comp_len = ro.completion_ids.shape
        if comp_len == 0:
            # No tokens — loss is 0 but we return a zero-filled telemetry dict
            # so downstream averaging keeps the same keys.
            zero = torch.zeros((), device=self.device, requires_grad=True)
            telemetry = {
                k: 0.0 for k in
                ("bgrpo/loss", "bgrpo/pg_loss", "bgrpo/kl", "bgrpo/clip_frac",
                 "bgrpo/mean_ratio", "bgrpo/log_ratio_std",
                 "bgrpo/policy_logprob_mean", "bgrpo/ref_logprob_mean",
                 "bgrpo/adv_mean", "bgrpo/adv_std", "bgrpo/n_tokens")
            }
            return zero, telemetry

        full_ids, prompt_len = self._build_full_ids(ro)
        # Teacher-forced alignment: logits at position t predict token t+1.
        # Completion targets live at full_ids[:, prompt_len:]; the logits
        # that predict them are at [prompt_len-1, prompt_len+comp_len-1).
        policy_logits, _ = forward_logits(self.model, full_ids)
        policy_completion_logits = policy_logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :]
        logp_policy = gather_token_logprobs(policy_completion_logits, ro.completion_ids)

        return self.objective.compute(
            logprobs_policy=logp_policy,
            logprobs_rollout=ro.rollout_logprobs.detach(),
            logprobs_reference=ref_logprobs.detach(),
            advantages=advantages.detach(),
            completion_mask=ro.completion_mask,
        )

    def _build_full_ids(self, ro: RolloutResult) -> tuple[torch.Tensor, int]:
        w = ro.completion_ids.size(0)
        prompt_repeat = ro.prompt_ids.expand(w, -1)
        return torch.cat([prompt_repeat, ro.completion_ids], dim=1), ro.prompt_ids.size(1)

    def _encode_prompt(self, prompt_str: str) -> torch.Tensor:
        """Tokenize the prompt half of a dataset line and append the MASK sentinel.

        Dataset lines are ``<expanded> <sep> <target>`` where <sep> is either
        ``?`` (extended vocab) or ``⁇`` (simple vocab). Both variants are
        supported; we fail loudly if the separator is missing entirely, since
        feeding the whole line including the target back to the model would
        silently train BGRPO on the answer.
        """
        mask_char = self.tokenizer.MASK_CHAR
        head = prompt_str.replace("?", mask_char).split(mask_char)[0]
        if head == prompt_str:
            raise ValueError(
                f"prompt missing separator '?' / '{mask_char}': {prompt_str!r}"
            )
        ids = self.tokenizer.encode(head)
        ids.append(self.tokenizer.mask_token_id)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    # --- logging / checkpointing --------------------------------------------

    def _log_step(self, tele: dict[str, float], correctness: dict, inner: int) -> None:
        n_total = max(correctness["n_total"], 1)
        payload = {
            **tele,
            "bgrpo/correct_frac": correctness["n_correct"] / n_total,
            "bgrpo/outer_iter": self.outer_iter,
            "bgrpo/inner_iter": inner,
        }
        if wandb.run is not None:
            wandb.log(payload, step=self.global_step)
        logger.info(
            "outer=%d inner=%d loss=%.4f kl=%.4f clip_frac=%.2f correct=%d/%d",
            self.outer_iter, inner,
            tele.get("bgrpo/loss", float("nan")),
            tele.get("bgrpo/kl", float("nan")),
            tele.get("bgrpo/clip_frac", float("nan")),
            correctness["n_correct"], correctness["n_total"],
        )

    def _save_checkpoint(self, tag: str) -> None:
        out = Path(self.cfg.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
