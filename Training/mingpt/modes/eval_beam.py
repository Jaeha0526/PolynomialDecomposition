"""
Beam-search evaluation (``debug_beam``).

Powers D4 beam-width scaling, D1/D2 beam numbers, and the simplification
leaf-count runs. For each test line, runs beam search with width
``args.beam_width`` and records the rank (1-indexed) at which the correct
answer appears, if any.
"""

import os

import torch
from tqdm import tqdm

try:
    from .. import dataset, utils
except ImportError:
    import dataset
    import utils


def run(args, gpt, chars_symbolic, device):
    block_size = args.block_size
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path), strict=False)

    tokentype = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )

    max_beam = args.beam_width
    beam_widths = list(range(1, max_beam + 1))
    correct_counts = {w: 0 for w in beam_widths}
    correct_idx = {w: [] for w in beam_widths}

    total = 0
    os.makedirs(os.path.dirname(args.outputs_path), exist_ok=True)

    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        idx = 0
        for i, line in tqdm(enumerate(open(args.evaluate_corpus_path, encoding="utf-8"))):
            if total == args.max_test:
                break

            line_here = line.replace("?", tokentype.MASK_CHAR)
            input_str = line_here.split(tokentype.MASK_CHAR)[0]

            pred_str, correct_beam_rank = utils.LLM_BeamSearch_check(
                gpt, input_str, tokentype, device, args
            )

            if correct_beam_rank != -1:
                for w in beam_widths:
                    if w >= correct_beam_rank:
                        correct_counts[w] += 1
                        correct_idx[w].append(idx)

            pred_output = (
                line_here.split(tokentype.MASK_CHAR)[1].replace(" ", "")
                + tokentype.MASK_CHAR
                + (pred_str if pred_str is not False else "False")
            )

            print(f"final pred : {input_str} -> {pred_str} \n", flush=True)
            fout.write(pred_output + "\n")
            fout.flush()

            total += 1

            print("\nCurrent Statistics:")
            for w in beam_widths:
                print(
                    f"Beam width {w}: {correct_counts[w]} out of {total}: "
                    f"{(correct_counts[w] / total * 100):.2f}%"
                )
            print("\n", flush=True)

            if i % 50 == 49:
                fout.write(f"Statistics at line {i} :\n")
                for w in beam_widths:
                    fout.write(
                        f"Beam width {w}: {correct_counts[w]} out of {total}: "
                        f"{(correct_counts[w] / total * 100):.2f}%\n"
                    )
                fout.write("\n")
                fout.flush()

            idx += 1

    if total > 0:
        print("\nFinal Statistics:")
        for w in beam_widths:
            print(
                f"Beam width {w}: {correct_counts[w]} out of {total}: "
                f"{(correct_counts[w] / total * 100):.2f}%"
            )
        print("\nCorrect Indices : width & indices")
        for w in beam_widths:
            print(f"{w} : \n {correct_idx[w]}")
    else:
        print(
            f"Predictions written to {args.outputs_path}; no targets provided",
            flush=True,
        )
