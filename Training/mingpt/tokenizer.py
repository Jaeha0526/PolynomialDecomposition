"""
HF-compatible tokenizer for the symbolic polynomial vocabulary.

Thin wrapper around a vocab list that exposes the subset of the HuggingFace
tokenizer API that TRL's GRPOTrainer (0.16.0) and our evaluation code expect:
``encode``, ``decode``, ``__call__``, ``batch_decode``, ``save_pretrained``,
plus ``pad_token_id`` / ``eos_token_id`` / ``mask_token_id`` attributes.

PAD is used as EOS and BOS because our training data is space-tokenized
with PAD-fill at the end, and we ignore PAD in the cross-entropy loss.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch

try:
    from .vocab import PAD_CHAR, MASK_CHAR
except ImportError:
    from vocab import PAD_CHAR, MASK_CHAR


class SymbolicTokenizer:
    def __init__(self, vocab: list[str]):
        if PAD_CHAR not in vocab:
            raise ValueError(f"Padding character {PAD_CHAR!r} missing from vocabulary")
        if MASK_CHAR not in vocab:
            raise ValueError(f"Mask character {MASK_CHAR!r} missing from vocabulary")

        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

        # Special-token attributes. PAD doubles as EOS because training
        # masks PAD positions in the loss and decoded strings are split
        # on MASK_CHAR, not EOS. END_INDEX / MASK_INDEX are the names
        # utils.beam_search / LLM_BeamSearch_check expect.
        #
        # BGRPO and TRL's GRPOTrainer read additional HF-style aliases
        # off the tokenizer (bos_token_id, etc.). Those are intentionally
        # NOT set here — BGRPO is slated for a redesign that will replace
        # the TRL coupling entirely; test_08_bgrpo_smoke is expected to
        # fail until that redesign lands.
        self.PAD_CHAR = PAD_CHAR
        self.MASK_CHAR = MASK_CHAR
        self.pad_token_id = self.stoi[PAD_CHAR]
        self.mask_token_id = self.stoi[MASK_CHAR]
        self.eos_token_id = self.pad_token_id
        self.END_INDEX = self.eos_token_id
        self.MASK_INDEX = self.mask_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Split on spaces, drop empty fragments, map to IDs. Unknown tokens are dropped."""
        return [self.stoi[t] for t in text.split(" ") if t and t in self.stoi]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        toks = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special_tokens and tid in (self.pad_token_id,):
                continue
            toks.append(self.itos.get(tid, ""))
        return " ".join(toks)

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **_) -> List[str]:
        if torch.is_tensor(sequences):
            sequences = sequences.tolist()
        # Preserve legacy behavior: include special tokens so callers can split on MASK_CHAR.
        return [self.decode(seq, skip_special_tokens=False) for seq in sequences]

    def __call__(
        self,
        text,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        **_,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        batch = [{"input_ids": self.encode(t)} for t in text]

        if truncation and max_length:
            for d in batch:
                d["input_ids"] = d["input_ids"][:max_length]

        for d in batch:
            d["attention_mask"] = [1] * len(d["input_ids"])

        if padding:
            target_len = max_length or max((len(d["input_ids"]) for d in batch), default=0)
            for d in batch:
                pad = target_len - len(d["input_ids"])
                d["input_ids"] = d["input_ids"] + [self.pad_token_id] * pad
                d["attention_mask"] = d["attention_mask"] + [0] * pad

        input_ids = [d["input_ids"] for d in batch]
        attention_mask = [d["attention_mask"] for d in batch]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        if return_tensors is None:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        raise ValueError(f"Unsupported return_tensors format: {return_tensors!r}")

    def save_pretrained(self, save_directory: str, **_):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
