"""
Classify vocabulary tokens into the categories used by the paper's Appendix C
analysis: SIGN (P/N), OPERATOR (+, *, ^), DELIMITER (&, ?, ⁇), NUMBER (digit
or multi-digit integer in the extended vocab), VARIABLE (a/b/x/y/..., a0..a18,
b0..b18, n1..n18), PAD (□), OTHER (anything unrecognised).

The category of a token is derived purely from its string form so the same
rules apply to both the 31-token simple vocab and the ~174-token extended
vocab built by ``mingpt.vocab``.
"""

from __future__ import annotations

import re
from enum import IntEnum

import torch

from mingpt.vocab import PAD_CHAR, MASK_CHAR


class TokenCategory(IntEnum):
    PAD = 0
    SIGN = 1
    OPERATOR = 2
    DELIMITER = 3
    NUMBER = 4
    VARIABLE = 5
    OTHER = 6


CATEGORY_NAMES: dict[int, str] = {c.value: c.name for c in TokenCategory}

_OPERATORS = {"+", "*", "^"}
_SIGNS = {"P", "N"}
_DELIMITERS = {"&", "?", MASK_CHAR}
_VARIABLE_RE = re.compile(r"^[abn]\d+$")   # a0, b18, n3, ...
_SHORT_VAR = set("abcdexyz")               # single-letter variables in simple vocab


def categorize(tok: str) -> TokenCategory:
    """Map a token string to its category."""
    if tok == PAD_CHAR:
        return TokenCategory.PAD
    if tok in _SIGNS:
        return TokenCategory.SIGN
    if tok in _OPERATORS:
        return TokenCategory.OPERATOR
    if tok in _DELIMITERS:
        return TokenCategory.DELIMITER
    if tok.isdigit():
        return TokenCategory.NUMBER
    if tok in _SHORT_VAR or _VARIABLE_RE.match(tok):
        return TokenCategory.VARIABLE
    return TokenCategory.OTHER


def category_map(itos: list[str]) -> torch.LongTensor:
    """Vector of shape (vocab_size,) with the integer category of each token id."""
    return torch.tensor([int(categorize(t)) for t in itos], dtype=torch.long)
