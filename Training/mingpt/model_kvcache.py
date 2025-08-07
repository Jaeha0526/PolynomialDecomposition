"""
GPT model with KV-cache support for efficient inference
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from model import GPT, GPTConfig, top_k_logits
from kv_cache_module import replace_with_kv_cache_attention


class GPTWithKVCache(GPT):
    """
    GPT model with KV-cache support for efficient autoregressive generation.
    Inherits from the original GPT but adds KV-cache functionality.
    """
    
    def __init__(self, config, use_flash_attention=True):
        super().__init__(config)
        # Replace attention blocks with KV-cache aware versions
        self = replace_with_kv_cache_attention(self, use_flash_attention=use_flash_attention)
        self.use_flash_attention = use_flash_attention
        print(f"GPTWithKVCache initialized with {'Flash' if use_flash_attention else 'Standard'} Attention + KV-Cache")
    
    def forward(self, idx, targets=None, past_key_values=None, use_cache=False, return_attentions=False, output_hidden_states=False):
        """
        Forward pass with KV-cache support.
        
        Args:
            idx: input token indices [batch, seq_len]
            targets: target indices for loss computation
            past_key_values: list of past (key, value) tuples for each layer
            use_cache: whether to return updated key-value pairs
            return_attentions: whether to return attention weights
            
        Returns:
            logits: output logits [batch, seq_len, vocab_size]
            loss: language modeling loss (if targets provided)
            present_key_values: updated cache (if use_cache=True)
        """
        device = idx.device
        b, t = idx.size()
        
        # When using cache for generation, we only process new tokens
        if past_key_values is not None and past_key_values[0] is not None:
            # Get the sequence length from the cache
            past_length = past_key_values[0][0].size(2)  # (batch, n_head, seq_len, head_dim)
            # Make sure we're not exceeding block size
            # Allow reprocessing last token for generation (past_length + 1 when t=1)
            assert past_length + t <= self.block_size + 1, \
                f"Cannot forward, sequence length {past_length + t} exceeds block size {self.block_size} + 1"
        else:
            past_length = 0
            assert t <= self.block_size, \
                f"Cannot forward, sequence length {t} exceeds block size {self.block_size}"
        
        # Token embeddings
        token_embeddings = self.tok_emb(idx)  # [batch, t, n_embd]
        
        # Position embeddings - need to use the right positions when using cache
        position_embeddings = self.pos_emb[:, past_length:past_length + t, :]  # [1, t, n_embd]
        
        x = self.drop(token_embeddings + position_embeddings)
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        
        # Forward through transformer blocks
        present_key_values = []
        all_attentions = [] if return_attentions else None
        
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            if return_attentions:
                x, present, attn_weights = block(x, layer_past=layer_past, use_cache=use_cache, return_attentions=True)
                all_attentions.append(attn_weights)
            else:
                x, present = block(x, layer_past=layer_past, use_cache=use_cache, return_attentions=False)
            
            if use_cache:
                present_key_values.append(present)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        
        # Store hidden states before projection if requested
        hidden_states = None
        if output_hidden_states:
            # x contains the final hidden states after layer norm
            hidden_states = (x,)  # TRL expects a tuple
        
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Return based on what was requested
        # Note: We need to maintain backward compatibility with existing code
        if output_hidden_states:
            # When hidden states are requested, return them in all cases
            if use_cache and return_attentions:
                return logits, loss, present_key_values, all_attentions, hidden_states
            elif use_cache:
                return logits, loss, present_key_values, hidden_states
            elif return_attentions:
                return logits, loss, all_attentions, hidden_states
            else:
                return logits, loss, hidden_states
        else:
            # Original return format when hidden states not requested
            if use_cache and return_attentions:
                return logits, loss, present_key_values, all_attentions
            elif use_cache:
                return logits, loss, present_key_values
            elif return_attentions:
                return logits, loss, all_attentions
            else:
                return logits, loss
    
    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Efficient generation using KV-cache.
        
        Args:
            idx: initial sequence [batch, seq_len]
            max_new_tokens: number of tokens to generate
            temperature: softmax temperature
            do_sample: whether to sample or use argmax
            top_k: if specified, only sample from top k tokens
            
        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Get input for this iteration
            if past_key_values is None:
                # First iteration: process all tokens
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
                logits, _, past_key_values = self.forward(idx_cond, use_cache=True)
                # Get logits for last position
                logits = logits[:, -1, :] / temperature
            else:
                # Subsequent iterations: only process the new token
                idx_cond = idx[:, -1:]  # Just the last token
                logits, _, past_key_values = self.forward(idx_cond, past_key_values=past_key_values, use_cache=True)
                logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @torch.no_grad()
    def beam_search_with_cache(self, x, steps, beam_width=3, temperature=1.0, pad_token=None, dataset=None):
        """
        Beam search with KV-cache for efficient generation.
        
        Each beam maintains its own KV-cache to avoid recomputation.
        """
        self.eval()
        device = x.device
        batch_size = x.size(0)
        
        # Initialize beams - start with single beam like original implementation
        # Process initial sequence once
        initial_logits, _, initial_cache = self.forward(x, use_cache=True)
        
        # Start with single beam, will expand in first step
        beams = [{
            'sequence': x.clone(),
            'cache': initial_cache,
            'score': 0.0,
            'finished': False
        }]
        
        for step in range(steps):
            all_candidates = []
            
            for beam in beams:
                if beam['finished']:
                    all_candidates.append(beam)
                    continue
                
                # For the first step, use the initial logits; for subsequent steps, call forward
                if step == 0:
                    # Use logits from initial forward pass
                    logits = initial_logits
                    new_cache = beam['cache']  # Cache is already set from initial pass
                else:
                    # Get next token predictions using cache
                    last_token = beam['sequence'][:, -1:]
                    logits, _, new_cache = self.forward(
                        last_token, 
                        past_key_values=beam['cache'], 
                        use_cache=True
                    )
                
                # Get probabilities for next tokens (match original implementation)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get top beam_width tokens
                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)
                
                # Create new candidates
                for k in range(beam_width):
                    next_token_id = topk_indices[:, k].unsqueeze(1)
                    
                    # Handle MASK_INDEX → END_INDEX conversion like original
                    if dataset and hasattr(dataset, 'MASK_INDEX') and next_token_id.item() == dataset.MASK_INDEX:
                        next_token_id[0, 0] = dataset.END_INDEX
                    
                    new_sequence = torch.cat([beam['sequence'], next_token_id], dim=1)
                    
                    # Deep copy the cache for each candidate to ensure independence
                    # Each candidate needs its own cache copy
                    candidate_cache = [(k.clone(), v.clone()) for k, v in new_cache] if new_cache else None
                    
                    # Use log probability for scoring like original
                    log_prob = torch.log(topk_probs[0, k]).item()
                    
                    candidate = {
                        'sequence': new_sequence,
                        'cache': candidate_cache,  # Each beam gets its own cache copy
                        'score': beam['score'] + log_prob,
                        'finished': (next_token_id.item() == dataset.END_INDEX if dataset and hasattr(dataset, 'END_INDEX') else False)
                    }
                    all_candidates.append(candidate)
            
            # Select top beam_width candidates
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = all_candidates[:beam_width]
            
            # Early stopping if all beams are finished
            if all(beam['finished'] for beam in beams):
                break
        
        # FIX: Pad sequences to the same length before stacking
        # This handles the case where different beams have different sequence lengths
        max_length = max(beam['sequence'].size(1) for beam in beams)
        
        padded_sequences = []
        for beam in beams:
            seq = beam['sequence']
            if seq.size(1) < max_length:
                # Pad with pad_token if provided, otherwise use 0
                padding_value = pad_token if pad_token is not None else 0
                padding = torch.full((seq.size(0), max_length - seq.size(1)), 
                                     padding_value, dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, padding], dim=1)
            padded_sequences.append(seq)
        
        # Stack padded sequences
        stacked = torch.stack(padded_sequences)
        if stacked.dim() == 3 and stacked.size(1) == 1:
            stacked = stacked.squeeze(1)
        return stacked


# Utility function to create KV-cache model from existing checkpoint
def create_kvcache_model_from_checkpoint(checkpoint_path, config, use_flash_attention=True):
    """
    Load a standard GPT checkpoint and convert it to KV-cache model.
    
    Args:
        checkpoint_path: path to saved model state dict
        config: GPTConfig object
        use_flash_attention: whether to use Flash Attention
        
    Returns:
        GPTWithKVCache model with loaded weights
    """
    # Create KV-cache model
    model = GPTWithKVCache(config, use_flash_attention=use_flash_attention)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # The state dict keys might not match exactly due to the module changes
    # We'll need to map them appropriately
    model.load_state_dict(state_dict, strict=False)
    
    return model