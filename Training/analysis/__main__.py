"""
CLI dispatcher: ``python -m analysis {confusion|circuits|viz} [flags]``.
Each subcommand re-delegates to its module's ``main``. Run with no subcommand
to see the list of available modes.
"""

from __future__ import annotations

import sys

MODES = {
    "confusion": "analysis.confusion",
    "circuits":  "analysis.circuits",
    "viz":       "analysis.viz",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m analysis {confusion|circuits|viz} [args...]")
        print("\nmodes:")
        for k in MODES:
            print(f"  {k}")
        sys.exit(0 if len(sys.argv) > 1 else 2)

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"unknown mode {mode!r}; choose one of {list(MODES)}", file=sys.stderr)
        sys.exit(2)

    import importlib
    m = importlib.import_module(MODES[mode])
    # Shift argv so argparse inside the subcommand sees a clean list.
    sys.argv = [f"analysis {mode}"] + sys.argv[2:]
    m.main()


if __name__ == "__main__":
    main()
