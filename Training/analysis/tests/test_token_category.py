"""Unit tests for token-category classification."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "Training") not in sys.path:
    sys.path.insert(0, str(REPO / "Training"))

from mingpt.vocab import MASK_CHAR, PAD_CHAR, build_extended_vocab, build_simple_vocab  # noqa: E402

from analysis.token_category import TokenCategory, categorize, category_map  # noqa: E402


def test_category_of_known_tokens():
    assert categorize("P") == TokenCategory.SIGN
    assert categorize("N") == TokenCategory.SIGN
    assert categorize("+") == TokenCategory.OPERATOR
    assert categorize("*") == TokenCategory.OPERATOR
    assert categorize("^") == TokenCategory.OPERATOR
    assert categorize("&") == TokenCategory.DELIMITER
    assert categorize("?") == TokenCategory.DELIMITER
    assert categorize(MASK_CHAR) == TokenCategory.DELIMITER
    assert categorize(PAD_CHAR) == TokenCategory.PAD
    assert categorize("7") == TokenCategory.NUMBER
    assert categorize("101") == TokenCategory.NUMBER
    assert categorize("a") == TokenCategory.VARIABLE
    assert categorize("a0") == TokenCategory.VARIABLE
    assert categorize("b18") == TokenCategory.VARIABLE
    assert categorize("n12") == TokenCategory.VARIABLE


def test_category_map_simple_vocab_covers_expected_counts():
    vocab = build_simple_vocab()
    cm = category_map(vocab)
    assert cm.shape[0] == len(vocab)
    # Simple vocab: 1 PAD, 2 sign, 3 op, 3 delim (MASK, ?, &), 10 digits,
    # plus variables (a,b,c,d,e,x,y,z + a0,a1,b0,b1 = 12).
    counts = {c.value: int((cm == c.value).sum()) for c in TokenCategory}
    assert counts[TokenCategory.PAD] == 1
    assert counts[TokenCategory.SIGN] == 2
    assert counts[TokenCategory.OPERATOR] == 3
    assert counts[TokenCategory.DELIMITER] == 3
    assert counts[TokenCategory.NUMBER] == 10
    assert counts[TokenCategory.VARIABLE] == 12
    # Nothing should fall into OTHER for a valid vocab.
    assert counts[TokenCategory.OTHER] == 0


def test_category_map_extended_vocab_has_all_categories():
    vocab = build_extended_vocab(max_number_token=101)
    cm = category_map(vocab)
    for c in (TokenCategory.PAD, TokenCategory.SIGN, TokenCategory.OPERATOR,
              TokenCategory.DELIMITER, TokenCategory.NUMBER, TokenCategory.VARIABLE):
        assert (cm == c.value).sum() > 0, f"missing category {c.name}"
    assert (cm == TokenCategory.OTHER.value).sum() == 0
