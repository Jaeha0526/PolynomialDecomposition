"""
KV-Cache implementation for minGPT
Provides drop-in replacement attention modules with KV-caching support
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttentionWithKVCache(nn.Module):
    """
    Multi-head self-attention with KV-cache support.
    Drop-in replacement for CausalSelfAttention that caches key/value pairs.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attention_weights = None  # For compatibility

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass with optional KV-caching.
        
        Args:
            x: input tensor [batch, seq_len, n_embd]
            layer_past: tuple of (past_key, past_value) or None
            use_cache: whether to return updated cache
            
        Returns:
            y: output tensor [batch, seq_len, n_embd]
            present: updated (key, value) cache if use_cache=True, else None
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        # Note: We always compute Q for all positions, but K,V only for new positions
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # If we have past key/values, concatenate them
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=2)  # (B, nh, T_past + T, hs)
            v = torch.cat([past_value, v], dim=2)  # (B, nh, T_past + T, hs)

        # Cache current key/values if requested
        present = (k, v) if use_cache else None

        # Causal self-attention
        # (B, nh, T, hs) x (B, nh, hs, T_past + T) -> (B, nh, T, T_past + T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        # When using cache, we need to be careful about the mask dimensions
        if layer_past is not None:
            # We're generating new tokens, need different masking
            # Only mask to prevent attending to future tokens in the NEW sequence
            att_mask = torch.ones(B, 1, T, k.size(2), device=x.device)
            # Can attend to all past tokens and current tokens
            # This naturally handles the causal mask for generation
        else:
            # Normal training/full sequence mode
            att_mask = self.mask[:, :, :T, :k.size(2)]
            att = att.masked_fill(att_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.attention_weights = att
        
        # (B, nh, T, T_past + T) x (B, nh, T_past + T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs

        # Output projection
        y = self.resid_drop(self.proj(y))
        
        return y, present


class FlashCausalSelfAttentionWithKVCache(nn.Module):
    """
    Flash Attention with KV-cache support.
    Uses PyTorch's scaled_dot_product_attention with KV-caching.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Separate projections to match original FlashCausalSelfAttention
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_dropout = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attention_weights = None  # For compatibility

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass with Flash Attention and optional KV-caching.
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        q = self.query(x)  # (B, T, n_embd)
        k = self.key(x)    # (B, T, n_embd)
        v = self.value(x)  # (B, T, n_embd)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # If we have past key/values, concatenate them
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=2)  # (B, nh, T_past + T, hs)
            v = torch.cat([past_value, v], dim=2)  # (B, nh, T_past + T, hs)

        # Cache current key/values if requested
        present = (k, v) if use_cache else None

        # Use Flash Attention
        # When using KV-cache, we can't use is_causal=True because that assumes square attention
        # Instead, we need to handle causality through attention mask
        if layer_past is not None:
            # For generation with cache, we can attend to all previous tokens
            # No mask needed - we can see all past and current tokens
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,  # We handle causality differently with cache
                scale=1.0 / math.sqrt(k.size(-1))
            )
        else:
            # Normal mode without cache - use is_causal for efficiency
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(k.size(-1))
            )
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.proj(y))
        
        return y, present


class BlockWithKVCache(nn.Module):
    """Transformer block with KV-cache support"""

    def __init__(self, config, use_flash_attention=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Choose attention implementation
        if use_flash_attention:
            self.attn = FlashCausalSelfAttentionWithKVCache(config)
        else:
            self.attn = CausalSelfAttentionWithKVCache(config)
            
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, use_cache=False, return_attentions=False):
        # Attention with residual connection
        attn_output, present = self.attn(self.ln1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_output
        
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        
        if return_attentions:
            return x, present, self.attn.attention_weights
        else:
            return x, present


def replace_with_kv_cache_attention(model, use_flash_attention=True):
    """
    Replace attention modules in a model with KV-cache aware versions.
    
    Args:
        model: GPT model to modify
        use_flash_attention: whether to use Flash Attention variant
        
    Returns:
        Modified model with KV-cache support
    """
    from model import Block
    
    # Get model device
    device = next(model.parameters()).device
    
    # Replace all blocks with KV-cache aware blocks
    new_blocks = nn.ModuleList()
    for i, block in enumerate(model.blocks):
        # Create new block with KV-cache support
        new_block = BlockWithKVCache(model.config, use_flash_attention=use_flash_attention).to(device)
        
        # Copy weights from original block
        # First check what type of attention the original block has
        orig_attn = block.attn
        new_attn = new_block.attn
        
        # Handle different attention types - now both should have separate QKV
        if hasattr(orig_attn, 'key') and hasattr(new_attn, 'key'):
            # Both have separate QKV
            new_attn.key.load_state_dict(orig_attn.key.state_dict())
            new_attn.query.load_state_dict(orig_attn.query.state_dict())
            new_attn.value.load_state_dict(orig_attn.value.state_dict())
        else:
            # Fallback for any other configuration
            print(f"Warning: Unexpected attention configuration in block {i}")
        
        # Copy output projection
        if hasattr(orig_attn, 'c_proj') and hasattr(new_attn, 'c_proj'):
            new_attn.c_proj.load_state_dict(orig_attn.c_proj.state_dict())
        elif hasattr(orig_attn, 'proj') and hasattr(new_attn, 'proj'):
            new_attn.proj.load_state_dict(orig_attn.proj.state_dict())
        elif hasattr(orig_attn, 'c_proj') and hasattr(new_attn, 'proj'):
            new_attn.proj.load_state_dict(orig_attn.c_proj.state_dict())
        elif hasattr(orig_attn, 'proj') and hasattr(new_attn, 'c_proj'):
            new_attn.c_proj.load_state_dict(orig_attn.proj.state_dict())
        
        # Copy layer norms
        new_block.ln1.load_state_dict(block.ln1.state_dict())
        new_block.ln2.load_state_dict(block.ln2.state_dict())
        
        # Copy MLP
        new_block.mlp.load_state_dict(block.mlp.state_dict())
        
        new_blocks.append(new_block)
    
    # Replace blocks in model
    model.blocks = new_blocks
    
    return model