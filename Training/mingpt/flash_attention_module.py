"""
Flash Attention drop-in replacement for CausalSelfAttention
Maintains exact same interface and behavior, just faster
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class FlashCausalSelfAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention using Flash Attention.
    Maintains exact same interface, attributes, and mathematical behavior.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads - SAME AS ORIGINAL
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization - SAME AS ORIGINAL
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection - SAME AS ORIGINAL
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask - kept for compatibility but not used in flash attention
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.attention_weights = None  # For compatibility
        
        # Store config for dropout probability
        self.attn_pdrop = config.attn_pdrop

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values - EXACT SAME COMPUTATION
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # Use Flash Attention via scaled_dot_product_attention
        # This replaces the manual attention computation
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # We use is_causal instead
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True,  # This handles the causal masking
            scale=1.0 / math.sqrt(k.size(-1))  # Same scaling as original
        )
        
        # Re-assemble all head outputs side by side - EXACT SAME AS ORIGINAL
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, nh, T, hs) -> (B, T, C)

        # Output projection - EXACT SAME AS ORIGINAL
        y = self.resid_drop(self.proj(y))
        return y


def replace_attention_with_flash_attention(model):
    """
    Replace all CausalSelfAttention modules in a model with FlashCausalSelfAttention.
    This modifies the model in-place.
    """
    from model import CausalSelfAttention
    
    # Get model device
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, CausalSelfAttention):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Create flash attention with same config and move to same device
            flash_attn = FlashCausalSelfAttention(model.config).to(device)
            
            # Copy weights from original attention
            flash_attn.key.load_state_dict(module.key.state_dict())
            flash_attn.query.load_state_dict(module.query.state_dict())
            flash_attn.value.load_state_dict(module.value.state_dict())
            flash_attn.proj.load_state_dict(module.proj.state_dict())
            
            # Replace the module
            setattr(parent, attr_name, flash_attn)
    
    return model