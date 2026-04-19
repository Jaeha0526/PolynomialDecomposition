"""
Unified CLI entry point for polynomial-decomposition training + evaluation.

Dispatches to one of the handlers in ``modes/`` based on the positional
``mode`` argument. The modes and the paper experiments they cover:

    inequality_finetune     — D1 / D2 / D3 supervised training
    inequality_evaluate4    — greedy evaluation (batched, length-grouped)
    debug_beam              — beam-search evaluation (D4, leaf-count)
    debug_multisampling     — multi-sample pass@k evaluation
    search_benchmark        — train/test overlap sanity check
    inequality_evaluate     — legacy unbatched greedy eval (kept for
                              backwards compatibility; no paper run uses it)

All model-shape flags (--block_size, --n_layer, --n_head, --n_embd,
--max_number_token, --extended_vocab) MUST match between training and
every downstream evaluation; they determine the tokenizer and the
checkpoint layout.

Originally forked from Andrej Karpathy's minGPT via Stanford CS224N A5.
"""

import argparse

import torch

import model
import utils
from flash_attention_module import replace_attention_with_flash_attention
from model_kvcache import GPTWithKVCache
from modes import DISPATCH
from vocab import build_vocab


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=sorted(DISPATCH.keys()),
                   help="Pipeline mode to run")
    p.add_argument("--char_corruption", action="store_true")
    p.add_argument("--reading_params_path", default=None)
    p.add_argument("--writing_params_path", default=None)
    p.add_argument("--pretrain_corpus_path", default=None)
    p.add_argument("--finetune_corpus_path", default=None)
    p.add_argument("--evaluate_corpus_path", default=None)
    p.add_argument("--valid_corpus_path", default=None)
    p.add_argument("--check_path", default="check.m")
    p.add_argument("--beam_width", default=5, type=int)
    p.add_argument("--max_test", default=3000, type=int)
    p.add_argument("--outputs_path", default=None)
    p.add_argument("--pretrain_lr", default=6e-3, type=float)
    p.add_argument("--finetune_lr", default=6e-4, type=float)
    p.add_argument("--lr_decay", default=1, type=int)
    p.add_argument("--shuffle", default=0, type=int)
    p.add_argument("--weight_decay", default=0.1, type=float)
    p.add_argument("--iteration_period", default=5000, type=int)
    p.add_argument("--num_epochs", default=3, type=int)
    p.add_argument("--block_size", default=128, type=int)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--evaluate_batch_size", default=32, type=int)
    p.add_argument("--dataset_name", default="inequality")
    p.add_argument("--exp_name", default="inequality")
    p.add_argument("--n_layer", default=4, type=int)
    p.add_argument("--n_head", default=8, type=int)
    p.add_argument("--n_embd", default=256, type=int)
    p.add_argument("--max_output_length", default=32, type=int)
    p.add_argument("--max_number_token", default=101, type=int)
    p.add_argument("--short_prediction", default=False)
    p.add_argument("--num_samples", type=int, default=30,
                   help="Number of samples for multisampling")
    p.add_argument("--sympy", default=0, type=int)
    p.add_argument("--test", default=0, type=int)
    p.add_argument(
        "--extended_vocab",
        action="store_true",
        help="Use extended vocabulary for multi-variable polynomial decomposition",
    )
    return p


def build_model(args, device, vocab_size, block_size):
    """Instantiate the model with attention optimized for this mode."""
    cfg = model.GPTConfig(
        vocab_size,
        block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    # Inference-only modes get the KV-cache model; training gets plain GPT +
    # Flash Attention (the cache is useless during teacher-forced training).
    if args.mode in {"inequality_evaluate4", "debug_beam"}:
        gpt = GPTWithKVCache(cfg, use_flash_attention=True).to(device)
    else:
        gpt = model.GPT(cfg).to(device)
        gpt = replace_attention_with_flash_attention(gpt)
    return gpt


def main() -> None:
    args = build_argparser().parse_args()
    # These flags are spelled as ints on the CLI for legacy compatibility.
    for name in ("lr_decay", "shuffle", "sympy", "test"):
        setattr(args, name, bool(getattr(args, name)))

    utils.set_seed(148)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chars_symbolic = build_vocab(
        extended=args.extended_vocab, max_number_token=args.max_number_token
    )
    if args.extended_vocab:
        print(
            f"Using extended vocabulary for multi-variable support "
            f"({len(chars_symbolic)} tokens); number tokens 0..{args.max_number_token - 1}"
        )
    else:
        print(f"Using simple vocabulary for single-variable ({len(chars_symbolic)} tokens)")

    print(f"block size: {args.block_size}")
    gpt = build_model(args, device, len(chars_symbolic), args.block_size)

    handler = DISPATCH[args.mode]
    handler(args, gpt, chars_symbolic, device)


if __name__ == "__main__":
    main()
