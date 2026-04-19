#!/usr/bin/env python3
"""
CLI entrypoint for the redesigned BGRPO trainer.

Usage:
    python run_bgrpo.py \\
        --model_name d2_arch_256_l6_best.pt \\
        --config_name d2_arch_256_l6.json \\
        --dataset_path ../../data_storage/things_on_paper/dataset/d2/training_dataset.txt \\
        --reward_type rank \\
        --output_dir outputs/bgrpo_d2_256_rank

Replaces the 5 legacy ``grpo_*.py`` scripts. See ``bgrpo/`` subpackage for
the implementation; this file is only argparse + glue.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

# Add project root so ``mingpt`` imports work when run as a script.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mingpt.model_loader import load_model_and_tokenizer  # noqa: E402
from mingpt.utils import validate_prediction_sympy  # noqa: E402

from bgrpo import (  # noqa: E402
    BGRPOConfig,
    BGRPOTrainer,
    ObjectiveConfig,
    RewardConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True,
                   help="SFT checkpoint filename under data_storage/model/")
    p.add_argument("--config_name", required=True,
                   help="JSON config filename under data_storage/model/model_configurations/")
    p.add_argument("--dataset_path", required=True,
                   help="Path to training prompts (one per line, '?-separated format)")
    p.add_argument("--output_dir", required=True)
    # Reward
    p.add_argument("--reward_type", choices=["simple", "rank", "reverse_rank"], default="rank")
    p.add_argument("--decay_base", type=int, default=None,
                   help="Rank decay base; defaults to --num_generations")
    p.add_argument("--wrong_penalty", type=float, default=0.0,
                   help="Reward for incorrect beams (paper: 0.0)")
    p.add_argument("--adjust_rewards", action="store_true",
                   help="Rescale wrong-rewards so sum(r)==0 per group")
    p.add_argument("--correct_reward", type=float, default=1.0)
    # Schedule
    p.add_argument("--num_generations", type=int, default=32, help="beam width")
    p.add_argument("--num_questions", type=int, default=8)
    p.add_argument("--num_iterations", type=int, default=5)
    p.add_argument("--total_training_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--rollout_temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None,
                   help="Restrict per-step sampling to top-k tokens (optional)")
    p.add_argument("--use_beam", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True,
                   help="true = BGRPO (beam rollout), false = GRPO (multinomial sampling)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--gradient_norm_clip", type=float, default=1.0)
    # Checkpointing / logging
    p.add_argument("--save_steps", type=int, default=5)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = PROJECT_ROOT.parent / "data_storage" / "model" / "model_configurations" / args.config_name
    model_dir_path = PROJECT_ROOT.parent / "data_storage" / "model"

    model, tokenizer = load_model_and_tokenizer(
        config_path=str(config_path),
        model_dir_path=str(model_dir_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        wrap_for_grpo=False,
        model_name=args.model_name,
        use_kvcache=False,  # we use plain forward during rollout
    )

    # Prompt file: one decomposition instance per line in the
    # ``<expanded> ? <target>`` format. We only feed the ``<expanded>?``
    # half to the model; ``<target>`` is discarded here but the sympy
    # validator parses ``<expanded>`` out of the same string.
    prompts = [ln.strip() for ln in open(args.dataset_path, encoding="utf-8") if ln.strip()]

    def correctness_fn(prompt_line: str, completion_str: str) -> bool:
        expanded = prompt_line.split("?")[0].strip()
        if not completion_str.strip():
            return False
        return validate_prediction_sympy(expanded, completion_str.strip())

    cfg = BGRPOConfig(
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        rollout_temperature=args.rollout_temperature,
        top_k=args.top_k,
        use_beam=args.use_beam,
        num_questions=args.num_questions,
        num_iterations=args.num_iterations,
        total_training_samples=args.total_training_samples,
        gradient_norm_clip=args.gradient_norm_clip,
        reward=RewardConfig(
            kind=args.reward_type,
            beam_width=args.num_generations,
            decay_base=args.decay_base,
            correct_reward=args.correct_reward,
            wrong_penalty=args.wrong_penalty,
            adjust_rewards=args.adjust_rewards,
        ),
        objective=ObjectiveConfig(
            clip_epsilon=args.clip_epsilon,
            kl_beta=args.kl_beta,
        ),
        save_steps=args.save_steps,
        output_dir=Path(args.output_dir),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={"model_name": args.model_name, "config_name": args.config_name},
    )

    trainer = BGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        correctness_fn=correctness_fn,
        config=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
