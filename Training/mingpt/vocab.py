"""
Canonical vocabulary builders for the polynomial-decomposition tokenizer.

Two vocabularies exist:
  * simple   — single-variable (numbers 0-9, variables a0/a1/b0/b1)
  * extended — multi-variable (numbers 0..max_number_token-1,
               variables a0..a18, b0..b18, n1..n18)

PAD_CHAR and MASK_CHAR are part of the vocabulary in slot 0 and slot 9
respectively for the simple vocab; their exact indices are derived from
``stoi`` at runtime, so callers should reference them by name, not position.
"""

PAD_CHAR = "\u25A1"  # □  empty square
MASK_CHAR = "\u2047"  # ⁇  double question mark


def build_simple_vocab() -> list[str]:
    """Single-variable vocabulary (~25 tokens)."""
    return [
        PAD_CHAR,
        "a", "b", "c", "d", "e", "x", "y", "z",
        MASK_CHAR, "?",
        "a0", "a1", "b0", "b1",
        "N", "P", "&", "+", "*", "^",
    ] + [str(i) for i in range(10)]


def build_extended_vocab(max_number_token: int) -> list[str]:
    """Multi-variable vocabulary with numbers 0..max_number_token-1."""
    variables = (
        [f"a{i}" for i in range(19)]
        + [f"b{i}" for i in range(19)]
        + [f"n{i}" for i in range(1, 19)]
    )
    return [
        PAD_CHAR,
        "a", "b", "c", "d", "e", "x", "y", "z",
        MASK_CHAR, "?",
        *variables,
        "N", "P", "&", "+", "*", "^",
    ] + [str(i) for i in range(max_number_token)]


def build_vocab(*, extended: bool, max_number_token: int = 101) -> list[str]:
    """Unified entry point. ``max_number_token`` is ignored for the simple vocab."""
    return build_extended_vocab(max_number_token) if extended else build_simple_vocab()
