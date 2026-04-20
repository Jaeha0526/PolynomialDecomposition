#!/bin/bash
# Run the analysis subpackage unit tests.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO/Training:${PYTHONPATH:-}"

python3 - <<'PY'
import sys, traceback
from analysis.tests import test_token_category, test_circuits
passed = failed = 0
for mod in (test_token_category, test_circuits):
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print(f"  ok  {mod.__name__}.{name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {mod.__name__}.{name}: {e}")
            traceback.print_exc()
            failed += 1
print(f"analysis unit: passed={passed} failed={failed}")
sys.exit(1 if failed else 0)
PY
