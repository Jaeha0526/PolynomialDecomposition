"""Private helpers shared across bgrpo submodules."""

import torch


def forward_logits(model, input_ids: torch.Tensor):
    """Call ``model.forward`` and return (logits, *rest*), tolerant of both
    plain-GPT ``(logits, loss)`` and HuggingFace ``CausalLMOutput``-style
    returns.
    """
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits, None
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None
