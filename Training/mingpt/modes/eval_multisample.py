"""
Multi-sample pass@k evaluation (``debug_multisampling``).

For each test line, generates ``args.num_samples`` independent samples from
the model and records the rank (0-indexed) at which the correct answer
first appears, if any. Used by BGRPO validation scripts to compute pass@k.
"""

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
    gpt.load_state_dict(torch.load(args.reading_params_path))

    tokentype = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )

    max_samples = args.num_samples if hasattr(args, "num_samples") else args.beam_width
    sample_widths = list(range(1, max_samples + 1))
    correct_counts = {w: 0 for w in sample_widths}
    correct_idx = {w: [] for w in sample_widths}

    total = 0

    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        idx = 0
        for i, line in tqdm(enumerate(open(args.evaluate_corpus_path, encoding="utf-8"))):
            if total == args.max_test:
                break

            line_here = line.replace("?", tokentype.MASK_CHAR)
            input_str = line_here.split(tokentype.MASK_CHAR)[0]

            pred_str, correct_sample_rank = utils.LLM_MultiSampling_check(
                gpt, input_str, tokentype, device, args
            )

            if correct_sample_rank != -1:
                for w in sample_widths:
                    if w > correct_sample_rank:
                        correct_counts[w] += 1
                        correct_idx[w].append(idx)

            pred_output = (
                line_here.split(tokentype.MASK_CHAR)[1].replace(" ", "")
                + tokentype.MASK_CHAR
                + (pred_str if pred_str != "False" else "False")
            )

            print(f"final pred : {input_str} -> {pred_str}", flush=True)
            fout.write(pred_output + "\n")
            fout.flush()

            total += 1

            print("\nCurrent Statistics:")
            for w in sample_widths:
                print(
                    f"Sample width {w}: {correct_counts[w]} out of {total}: "
                    f"{(correct_counts[w] / total * 100):.2f}%"
                )
            print("\n", flush=True)

            if i % 50 == 49:
                fout.write(f"Statistics at line {i} :\n")
                for w in sample_widths:
                    fout.write(
                        f"Sample width {w}: {correct_counts[w]} out of {total}: "
                        f"{(correct_counts[w] / total * 100):.2f}%\n"
                    )
                fout.write("\n")
                fout.flush()

            idx += 1

    if total > 0:
        print("Final Statistics:")
        for w in sample_widths:
            print(
                f"Sample width {w}: {correct_counts[w]} out of {total}: "
                f"{(correct_counts[w] / total * 100):.2f}%"
            )
    else:
        print(f"Predictions written to {args.outputs_path}; no targets provided")
