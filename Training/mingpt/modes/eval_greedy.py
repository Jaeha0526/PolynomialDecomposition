"""
Greedy evaluation modes.

``run_batched`` (``inequality_evaluate4``): the canonical greedy eval —
groups test lines by actual tokenized length and runs them in padding-free
batches. This is what powers D1 / D2 / D3 greedy accuracy numbers.

``run_unbatched`` (``inequality_evaluate``): the legacy one-at-a-time path.
Kept only because a handful of old shell scripts may still invoke it; no
paper experiment is tied to this path specifically.
"""

import os
from itertools import groupby

import torch
from tqdm import tqdm

try:
    from .. import dataset, utils
except ImportError:
    import dataset
    import utils


# --- helpers for the batched path ---------------------------------------------


def _actual_tensor_length(line: str) -> int:
    """Replicate the length computed by the on-the-fly tokenization below."""
    prompt = line.replace("?", "\u2047").split("\u2047")[0]
    tokens = [t for t in prompt.split(" ") if t]
    tokens.append("\u2047")
    return len(tokens)


def _group_lines_by_exact_length(lines):
    indexed = [(i, ln, _actual_tensor_length(ln)) for i, ln in enumerate(lines)]
    indexed.sort(key=lambda x: x[2])
    return [[(i, ln) for i, ln, _ in grp] for _, grp in groupby(indexed, key=lambda x: x[2])]


# --- run_batched: inequality_evaluate4 ----------------------------------------


def run_batched(args, gpt, chars_symbolic, block_size, device):
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None

    output_dir = os.path.dirname(args.outputs_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    gpt.load_state_dict(torch.load(args.reading_params_path), strict=False)

    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )

    lines = open(args.evaluate_corpus_path, encoding="utf-8").readlines()
    lines = lines[: args.max_test]
    grouped = _group_lines_by_exact_length(lines)

    predictions_dict = {}
    true_output_dict = {}
    batch_size = args.evaluate_batch_size

    for line_group in tqdm(grouped):
        for i in range(0, len(line_group), batch_size):
            batch_lines = line_group[i : i + batch_size]

            x_batch = []
            original_indices = []
            for original_index, line in batch_lines:
                prompt = line.replace("?", "\u2047").split("\u2047")[0]
                tokens = [t for t in prompt.split(" ") if t]
                tokens.append("\u2047")
                x_tensor = torch.tensor(
                    [test_dataset.stoi[s] for s in tokens], dtype=torch.long
                ).to(device)
                x_batch.append(x_tensor)
                original_indices.append(original_index)

            x_batch = torch.stack(x_batch)
            batch_preds = utils.sample(gpt, x_batch, args.max_output_length, sample=False)

            for j, pred in enumerate(batch_preds):
                completion = "".join([test_dataset.itos[int(k)] + " " for k in pred])

                if "\u2047" in completion:
                    pred2 = completion.split("\u2047")[1].strip()
                    if "\u2047" in pred2:
                        pred2 = pred2.split("\u2047")[0].strip()
                    if "\u25A1" in pred2:
                        pred2 = pred2.split("\u25A1")[0].strip()
                    pred_str = pred2.replace(" ", "")
                else:
                    pred2 = ""
                    pred_str = ""

                predictions_dict[original_indices[j]] = pred2
                line_here = batch_lines[j][1].replace("?", "\u2047")
                true_output_dict[original_indices[j]] = (
                    line_here.split("\u2047")[1] + " \u2047 " + pred2
                )

    sorted_indices = sorted(predictions_dict.keys())

    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        for i in sorted_indices:
            fout.write(true_output_dict[i] + "\n")

    predictions = [predictions_dict[i] for i in sorted_indices]
    total, correct = utils.evaluate_substitutions(
        args.evaluate_corpus_path, predictions, args.sympy
    )

    if total > 0:
        print(f"Correct: {correct} out of {total}: {correct / total * 100:.2f}%")
    else:
        print(f"Predictions written to {args.outputs_path}; no targets provided")


# --- run_unbatched: inequality_evaluate (legacy) ------------------------------


def run_unbatched(args, gpt, chars_symbolic, block_size, device):
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.evaluate_corpus_path is not None
    gpt.load_state_dict(torch.load(args.reading_params_path))

    test_dataset = dataset.SymbolicDataset(
        block_size,
        chars_symbolic,
        open(args.evaluate_corpus_path, encoding="utf-8").read(),
        use_extended_vocab=args.extended_vocab,
    )

    predictions = []
    with open(args.outputs_path, "w", encoding="utf-8") as fout:
        for line in tqdm(open(args.evaluate_corpus_path, encoding="utf-8")):
            line_here = line.replace("?", "\u2047")
            prompt = line_here.split("\u2047")[0]
            tokens = [t for t in prompt.split(" ") if t]
            tokens.append("\u2047")
            x = torch.tensor(
                [test_dataset.stoi[s] for s in tokens], dtype=torch.long
            )[None, ...].to(device)

            pred = utils.sample(gpt, x, args.max_output_length, sample=False)[0]
            completion = "".join([test_dataset.itos[int(i)] + " " for i in pred])
            pred = completion.replace(" ", "").split("\u2047")[1]
            pred2 = completion.split("\u2047")[1]
            predictions.append(pred)
            true_pred = line_here.split("\u2047")[1].replace(" ", "") + "\u2047" + pred2
            fout.write(true_pred + "\n")

        total, correct = utils.evaluate_substitutions(args.evaluate_corpus_path, predictions)

    if total > 0:
        print(f"Correct: {correct} out of {total}: {correct / total * 100}%")
    else:
        print(f"Predictions written to {args.outputs_path}; no targets provided")
