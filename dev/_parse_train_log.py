"""
Helper used by ``snapshot_eval_one_pass.sh`` and ``plot_snapshot_history.py``:
parse a training SLURM log and return the (global_iter, valid_loss) at the
most recent ``epoch N iter M`` line.

``global_iter`` is cumulative across epochs: if epoch 1 had 9987 iterations
and we're at epoch 2 iter 5000, global_iter == 9987 + 1 + 5000 == 14988.

Usage:
    python3 _parse_train_log.py <path-to-slurm.out>
prints ``<global_iter> <valid_loss>`` on stdout (or ``0 NaN`` if the log
contains no parseable iter lines yet).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PATTERN = re.compile(
    r"epoch (\d+) iter (\d+): train loss (\d+\.\d+) valid_loss (\d+\.\d+)"
)


def parse(path: Path) -> tuple[int, float]:
    if not path.exists():
        return 0, float("nan")
    text = path.read_text(errors="ignore").replace("\r", "\n")
    epoch_max_iter: dict[int, int] = {}
    latest: tuple[int, int, float, float] | None = None
    for m in _PATTERN.finditer(text):
        e = int(m.group(1))
        i = int(m.group(2))
        v = float(m.group(4))
        if i > epoch_max_iter.get(e, -1):
            epoch_max_iter[e] = i
        latest = (e, i, float(m.group(3)), v)

    if latest is None:
        return 0, float("nan")
    cur_epoch, cur_iter, _train, valid = latest
    # +1 per completed epoch because iter-N-of-epoch-K uses the position
    # after N iterations; global counts every step.
    global_iter = cur_iter + sum(
        epoch_max_iter[k] + 1 for k in range(1, cur_epoch) if k in epoch_max_iter
    )
    return global_iter, valid


def iter_timeline(path: Path) -> list[tuple[int, float, float]]:
    """Return [(global_iter, train_loss, valid_loss), ...] for every line."""
    if not path.exists():
        return []
    text = path.read_text(errors="ignore").replace("\r", "\n")

    # First pass: figure out epoch lengths.
    epoch_max_iter: dict[int, int] = {}
    for m in _PATTERN.finditer(text):
        e, i = int(m.group(1)), int(m.group(2))
        if i > epoch_max_iter.get(e, -1):
            epoch_max_iter[e] = i

    # Second pass: emit the timeline (deduped + sorted).
    seen: dict[tuple[int, int], tuple[float, float]] = {}
    for m in _PATTERN.finditer(text):
        e, i = int(m.group(1)), int(m.group(2))
        t, v = float(m.group(3)), float(m.group(4))
        seen[(e, i)] = (t, v)

    out: list[tuple[int, float, float]] = []
    for (e, i), (t, v) in sorted(seen.items()):
        global_iter = i + sum(epoch_max_iter[k] + 1 for k in range(1, e) if k in epoch_max_iter)
        out.append((global_iter, t, v))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: _parse_train_log.py <slurm.out>", file=sys.stderr)
        sys.exit(2)
    global_iter, valid_loss = parse(Path(sys.argv[1]))
    print(f"{global_iter} {valid_loss:.11f}")
