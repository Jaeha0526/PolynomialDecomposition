"""
Dispatchable per-mode handlers for ``run.py``.

Each module exposes ``run(args, gpt, chars_symbolic, device)``. The dispatch
table below maps mode names (as used on the command line and throughout
``things_on_paper/``) to the handler function.
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
    "inequality_evaluate4": eval_greedy.run_batched,
    "debug_beam": eval_beam.run,
    "search_benchmark": search_benchmark.run,
    "debug_multisampling": eval_multisample.run,
}
