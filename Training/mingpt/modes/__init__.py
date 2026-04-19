"""
Dispatchable per-mode handlers for ``run.py``.

Each module exposes ``run(args, gpt, chars_symbolic, block_size, device)``.
The dispatch table below maps mode names (as used on the command line and
throughout ``things_on_paper/exp*.sh``) to the handler function.
"""

from . import (
    finetune,
    eval_greedy,
    eval_beam,
    eval_multisample,
    search_benchmark,
)

DISPATCH = {
    "inequality_finetune": finetune.run,
    "inequality_evaluate": eval_greedy.run_unbatched,   # legacy, unbatched
    "inequality_evaluate4": eval_greedy.run_batched,    # canonical greedy eval
    "debug_beam": eval_beam.run,
    "search_benchmark": search_benchmark.run,
    "debug_multisampling": eval_multisample.run,
}
