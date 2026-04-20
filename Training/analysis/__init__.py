"""
Analysis tools for the polynomial-decomposition transformer.

Reproduces the two analyses from the paper's appendices:

* ``token_category`` / ``confusion`` — Appendix C, Table 4: per-token-category
  probability and accuracy (SIGN tokens are near-random; operators/numbers
  confident). This is the quantitative justification for beam search.
* ``circuits`` — Appendix D, "Monomial Heads": identify layer-0 previous-
  token heads (attend 1-5 back) and layer-1 within-monomial / delimiter
  heads proposed by the paper.

``viz`` renders Fig 9 (top-3 probability) and Fig 10 (attention heatmap);
``__main__`` exposes a CLI: ``python -m analysis {confusion|circuits|viz} ...``.
"""

from .token_category import (
    TokenCategory,
    categorize,
    category_map,
    CATEGORY_NAMES,
)

__all__ = [
    "TokenCategory",
    "categorize",
    "category_map",
    "CATEGORY_NAMES",
]
