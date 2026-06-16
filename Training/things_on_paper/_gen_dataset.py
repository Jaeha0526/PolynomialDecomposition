"""
Driver called from ``gen_dataset.sh``. Reads its config from environment
variables (populated by the shell wrapper) and invokes the sympy generator.

Kept as a proper .py file (not a heredoc) so multiprocessing's ``spawn``
start method can re-import it in worker processes. With a heredoc driven by
``python3 -``, ``spawn`` fails at ``runpy.run_path('<stdin>')``.
"""

import os
import sys
from pathlib import Path

REPO = Path(os.environ["REPO"]).resolve()
sys.path.insert(0, str(REPO))

from Data_Generation.Using_Sympy.using_sympy import generate_multivariate_datasets_parallel  # noqa: E402


def _int(name: str) -> int:
    return int(os.environ[name])


def main() -> None:
    out_dir = REPO / "data_storage" / "things_on_paper" / "dataset" / os.environ["DATA_TAG"]
    generate_multivariate_datasets_parallel(
        file_directory=str(out_dir),
        num_inner_vars=_int("NUM_INNER_VARS"),
        num_outer_vars=_int("NUM_OUTER_VARS"),
        max_degree_inner=_int("MAX_DEGREE_INNER"),
        max_degree_outer=_int("MAX_DEGREE_OUTER"),
        coeff_range_inner=(_int("COEFF_MIN_INNER"), _int("COEFF_MAX_INNER")),
        coeff_range_outer=(_int("COEFF_MIN_OUTER"), _int("COEFF_MAX_OUTER")),
        max_terms_inner=_int("MAX_TERMS_INNER"),
        max_terms_outer=_int("MAX_TERMS_OUTER"),
        num_train=_int("NUM_TRAIN"),
        num_test=_int("NUM_TEST"),
        num_valid=_int("NUM_VALID"),
        num_cpus=_int("WORKERS"),
    )
    print(f"Dataset written to {out_dir}")


if __name__ == "__main__":
    main()
