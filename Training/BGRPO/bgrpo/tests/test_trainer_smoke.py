"""
Integration smoke test for BGRPOTrainer.

Builds a tiny GPT + tokenizer in-process and runs 2 outer iterations × 2
inner policy updates. Passes if:
  * Training loop completes without error.
  * Parameters actually changed (gradient-step sanity).
  * Correctness-function callbacks were invoked.

Does NOT exercise the sympy polynomial validator — we use a trivial
``correctness_fn`` that returns True half the time so reward+advantage
are non-trivial.
"""

import sys
from pathlib import Path

import torch

# Reach into the mingpt package for model + tokenizer construction.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "Training") not in sys.path:
    sys.path.insert(0, str(REPO / "Training"))

from mingpt import model as mingpt_model  # noqa: E402
from mingpt.tokenizer import SymbolicTokenizer  # noqa: E402
from mingpt.vocab import build_simple_vocab  # noqa: E402

from bgrpo import (  # noqa: E402
    BGRPOConfig,
    BGRPOTrainer,
    ObjectiveConfig,
    RewardConfig,
)


def _build_tiny_model():
    vocab = build_simple_vocab()
    tokenizer = SymbolicTokenizer(vocab)
    config = mingpt_model.GPTConfig(
        vocab_size=len(vocab),
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = mingpt_model.GPT(config)
    model.END_INDEX = tokenizer.eos_token_id
    model.MASK_INDEX = tokenizer.mask_token_id
    return model, tokenizer


def test_bgrpo_trainer_smoke(tmp_path=None):
    """Full trainer loop executes, parameters change, no exceptions."""
    torch.manual_seed(0)
    model, tokenizer = _build_tiny_model()

    # A couple of dummy prompt lines in the dataset format.
    prompts = [
        "+ P 1 a0 ? + P 1 a0",
        "* P 2 a0 ? * P 2 a0",
        "+ * P 1 a0 P 3 ? + * P 1 a0 P 3",
        "+ a0 P 1 ? + a0 P 1",
    ]

    # Alternating correctness: first beam right, rest wrong.
    call_count = {"n": 0}
    def correctness_fn(prompt: str, completion: str) -> bool:
        call_count["n"] += 1
        return call_count["n"] % 4 == 1  # 25% correct rate

    cfg = BGRPOConfig(
        learning_rate=1e-4,  # higher than paper so smoke-scale updates are visible
        num_generations=4,
        max_new_tokens=8,
        num_questions=2,
        num_iterations=2,
        total_training_samples=4,  # 2 outer iterations
        save_steps=1000,           # effectively disabled
        reward=RewardConfig(kind="rank", beam_width=4, decay_base=4),
        objective=ObjectiveConfig(kl_beta=0.01, clip_epsilon=0.2),
        output_dir=None,
        wandb_project=None,
    )

    # Snapshot params before training.
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    trainer = BGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        correctness_fn=correctness_fn,
        config=cfg,
        device=torch.device("cpu"),
    )
    trainer.train()

    # Params should have changed (at least some).
    any_changed = False
    for n, p in model.named_parameters():
        if n in before and not torch.allclose(before[n], p.detach(), atol=0):
            any_changed = True
            break
    assert any_changed, "no parameters changed — trainer didn't actually step"
    assert call_count["n"] > 0, "correctness_fn was never called"
    # 2 outer iterations × num_questions=2 prompts × beam_width=4 = 16 calls minimum.
    assert call_count["n"] >= 2 * 2 * 4, f"too few correctness calls: {call_count['n']}"


if __name__ == "__main__":
    test_bgrpo_trainer_smoke()
    print("smoke test PASSED")
