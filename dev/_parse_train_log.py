"""
Helper used by ``snapshot_eval_one_pass.sh`` and ``plot_snapshot_history.py``:
parse one or more training SLURM logs (for chained continuations) and return
the most-recent ``(global_iter, valid_loss)``.

``global_iter`` is cumulative across epochs AND files. Within a single log,
epochs are offset by ``epoch_max_iter[k] + 1`` per completed epoch; across
files (chain continuations) each later file is further offset by the total
iteration count of earlier files. This way a monitor plot that spans an
original 5-epoch run + a chained 7-epoch resume shows a single contiguous
curve.

Usage:
    python3 _parse_train_log.py <slurm.out> [<slurm_2.out> ...]
prints ``<global_iter> <valid_loss>`` on stdout (or ``0 NaN`` if no log
in the chain has parseable iter lines yet).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PATTERN = re.compile(
    r"epoch (\d+) iter (\d+): train loss (\d+\.\d+) valid_loss (\d+\.\d+)"
)


def _file_timeline(path: Path) -> tuple[list[tuple[int, int, float, float]], int]:
    """Parse one slurm file. Returns (events, iters_in_file) where events is
    a list of (epoch, iter_within_epoch, train_loss, valid_loss) sorted, and
    iters_in_file is the total iterations the file covered (sum of per-epoch
    max_iter + 1) — used as the offset for the next chained file."""
    if not path.exists():
        return [], 0
    text = path.read_text(errors="ignore").replace("\r", "\n")
    epoch_max: dict[int, int] = {}
    seen: dict[tuple[int, int], tuple[float, float]] = {}
    for m in _PATTERN.finditer(text):
        e, i = int(m.group(1)), int(m.group(2))
        t, v = float(m.group(3)), float(m.group(4))
        seen[(e, i)] = (t, v)
        if i > epoch_max.get(e, -1):
            epoch_max[e] = i
    if not seen:
        return [], 0
    events = [(e, i, t, v) for (e, i), (t, v) in sorted(seen.items())]
    total_iters = sum(epoch_max[k] + 1 for k in epoch_max)
    return events, total_iters


def iter_timeline(*paths: Path) -> list[tuple[int, float, float]]:
    """Return [(global_iter, train_loss, valid_loss), ...] across any number
    of chained slurm log files. Each later file is offset by the total
    iteration count of earlier files."""
    out: list[tuple[int, float, float]] = []
    cum_offset = 0
    for p in paths:
        events, total = _file_timeline(Path(p))
        if not events:
            continue
        epoch_max: dict[int, int] = {}
        for e, i, _, _ in events:
            epoch_max[e] = max(epoch_max.get(e, -1), i)
        for e, i, t, v in events:
            local = i + sum(
                epoch_max[k] + 1 for k in range(1, e) if k in epoch_max
            )
            out.append((cum_offset + local, t, v))
        cum_offset += total
    return out


def parse(*paths: Path) -> tuple[int, float]:
    """``(global_iter, latest_valid_loss)`` across a chain of log files."""
    tl = iter_timeline(*paths)
    if not tl:
        return 0, float("nan")
    return tl[-1][0], tl[-1][2]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: _parse_train_log.py <slurm.out> [<slurm2.out> ...]",
              file=sys.stderr)
        sys.exit(2)
    global_iter, valid_loss = parse(*[Path(p) for p in sys.argv[1:]])
    print(f"{global_iter} {valid_loss:.11f}")
