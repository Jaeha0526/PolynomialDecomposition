"""
GPT model with Flash Attention implementation
Based on the original model.py but uses PyTorch's scaled_dot_product_attention
which automatically uses Flash Attention when available
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

torch.manual_seed(1)

# Import the original model components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))
from model import top_k_logits, Block, GPTConfig, GPT, GPT_hf


class FlashCausalSelfAttention(nn.Module):
    """
    Multi-head self-attention using Flash Attention via scaled_dot_product_attention
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attention_weights = None  # For compatibility

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Use PyTorch's scaled_dot_product_attention
        # This automatically uses Flash Attention when available
        # is_causal=True ensures proper masking for autoregressive models
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # We use is_causal instead
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(k.size(-1))
        )
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FlashBlock(nn.Module):
    """Transformer block with Flash Attention"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = FlashCausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, return_attentions=False):
        # Note: Flash Attention doesn't easily expose attention weights
        # so we ignore return_attentions for now
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if return_attentions:
            return x, None  # No attention weights with Flash Attention
        else:
            return x


class FlashGPT(GPT):
    """GPT model using Flash Attention"""
    
    def __init__(self, config):
        # Initialize parent but we'll replace the blocks
        super().__init__(config)
        
        # Replace standard blocks with Flash Attention blocks
        self.blocks = nn.ModuleList([FlashBlock(config) for _ in range(config.n_layer)])
        
        # Re-apply weight initialization
        self.apply(self._init_weights)
        
        print(f"FlashGPT initialized with {sum(p.numel() for p in self.parameters())} parameters")
        print("Using Flash Attention via scaled_dot_product_attention")


class FlashGPT_hf(FlashGPT):
    """Hugging Face compatible FlashGPT"""
    
    def __init__(self, config):
        super().__init__(config)
        # Add attributes expected by TRL/HF trainers
        self.warnings_issued = getattr(super(), 'warnings_issued', {})
        self.is_peft_model = False
        self.hf = True
        
    def add_model_tags(self, *args, **kwargs):
        """Dummy method for compatibility with GRPOTrainer."""
        pass

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, targets=None, return_attentions=False, **kwargs):
        """
        Overrides the forward method to map HF keyword args to the base GPT's positional args.
        """
        # Map input_ids keyword argument to idx positional argument
        idx = input_ids 
        
        # Call the original GPT forward method
        if return_attentions:
            logits, loss, attentions = super().forward(idx=idx, targets=targets, return_attentions=True)
        else:
            logits, loss = super().forward(idx=idx, targets=targets, return_attentions=False)
            attentions = None
        
        # Return the output in the standard Hugging Face format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=attentions,
        )