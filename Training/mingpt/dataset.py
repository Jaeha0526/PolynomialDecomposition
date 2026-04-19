"""
Symbolic dataset for polynomial-decomposition training.

Forked from Andrej Karpathy's minGPT via Stanford CS224N A5.
The only dataset actually used by this project is SymbolicDataset.
"""

import torch
from torch.utils.data import Dataset


class SymbolicDataset(Dataset):
    """
    One training pair per line:

        <expanded> ⁇ <outer> & <inner0> [& <inner1> ...]

    Raw files may use '?' as the separator; this class normalizes it to '⁇'
    unless ``use_extended_vocab`` is True (multi-variable mode keeps '?').
    """

    MASK_CHAR = "\u2047"  # ⁇  double-question-mark: mask / prompt-answer separator
    PAD_CHAR = "\u25A1"   # □  empty square: padding token

    def __init__(self, block_size, chars_symbolic, data, use_extended_vocab=False):
        self.block_size = block_size
        self.use_extended_vocab = use_extended_vocab

        self.stoi = {ch: i for i, ch in enumerate(chars_symbolic)}
        self.itos = {i: ch for i, ch in enumerate(chars_symbolic)}

        self.END_INDEX = self.stoi[self.PAD_CHAR]
        self.MASK_INDEX = self.stoi[self.MASK_CHAR]

        self.vocab_size = len(chars_symbolic)
        raw_lines = data.split("\n")

        # Drop lines that would overflow block_size. The tokenized layout is
        # ``<prompt> MASK <answer> MASK PAD* ``; count with that in mind
        # (+2 for the two MASK tokens), matching __getitem__.
        kept: list[str] = []
        dropped = 0
        for line in raw_lines:
            if not line:
                continue
            tokens_needed = self._tokens_needed(line)
            if tokens_needed > block_size:
                dropped += 1
                continue
            kept.append(line)

        if dropped:
            print(
                f"SymbolicDataset: dropped {dropped} / {len(raw_lines)} lines "
                f"(>{block_size} tokens) — bump --block_size if this is more than a rounding artifact."
            )

        self.data = kept
        print("data has %d characters, %d unique." % (len(data), self.vocab_size))

    def _tokens_needed(self, line: str) -> int:
        """How many tokens __getitem__ will produce for this raw line (pre-pad)."""
        if self.use_extended_vocab:
            sep = "?" if "?" in line else self.MASK_CHAR
            parts = line.split(sep)
        else:
            parts = line.replace("?", self.MASK_CHAR).split(self.MASK_CHAR)
        if len(parts) < 2:
            return self.block_size + 1  # malformed; drop
        inp_tokens = [t for t in parts[0].split(" ") if t]
        oup_tokens = [t for t in parts[1].split(" ") if t]
        # __getitem__ produces: inp + MASK + oup + MASK (+ PAD fill)
        return len(inp_tokens) + 1 + len(oup_tokens) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_here = self.data[idx]

        # Multi-variable uses '?' as separator; single-variable uses '⁇'.
        if self.use_extended_vocab:
            sep = "?" if "?" in data_here else self.MASK_CHAR
            inp, oup = data_here.split(sep)
        else:
            data_here = data_here.replace("?", self.MASK_CHAR)
            inp, oup = data_here.split(self.MASK_CHAR)

        inp = [t for t in inp.split(" ") if t]
        oup = [t for t in oup.split(" ") if t]
        inp.append(self.MASK_CHAR)
        x = inp + oup
        x.append(self.MASK_CHAR)
        pad_count = self.block_size - len(x)
        assert pad_count >= 0, (
            f"tokenized sequence length {len(x)} exceeds block_size {self.block_size}; "
            "__init__ should have filtered this line"
        )
        x.extend([self.PAD_CHAR] * pad_count)
        # Teacher-forcing targets: mask the prompt portion with PAD so it's ignored.
        y = [self.PAD_CHAR] * (len(inp) - 1) + x[len(inp):] + [self.PAD_CHAR]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y
