"""
Sanity-check mode: counts how many test-set lines appear verbatim in the
training corpus. Used to verify train/test overlap is negligible.
"""

from tqdm import tqdm


def run(args, gpt, chars_symbolic, device):
    assert args.finetune_corpus_path is not None
    assert args.evaluate_corpus_path is not None

    train_data_set = set(
        open(args.finetune_corpus_path, encoding="utf-8").read().split("\n")
    )
    print(f"train data has {len(train_data_set)} kinds of cases")

    test_data = open(args.evaluate_corpus_path, encoding="utf-8").read().split("\n")

    correct = 0
    total = 0
    for line in tqdm(test_data):
        if line in train_data_set:
            correct += 1
        total += 1

    print(f"Correct: {correct} out of {total}: {correct / total * 100}%")
