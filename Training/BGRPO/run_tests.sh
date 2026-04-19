#!/bin/bash
# Run all bgrpo/ tests. Usable both in .venv and under SLURM.
# Unit tests are pure CPU; integration smoke builds a tiny GPT.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

source "$REPO/.venv/bin/activate"
export PYTHONPATH="$HERE:$REPO/Training:${PYTHONPATH:-}"

python3 - <<'PY'
import sys, traceback
from bgrpo.tests import test_reward, test_objective, test_rollout
passed = failed = 0
for mod in (test_reward, test_objective, test_rollout):
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
print(f"unit: passed={passed} failed={failed}")
if failed:
    sys.exit(1)
PY

python3 "$HERE/bgrpo/tests/test_trainer_smoke.py"
echo "all bgrpo tests passed"
