"""
GPT_hf with Flash Attention and KV-Cache support for BGRPO
Combines HuggingFace compatibility, Flash Attention, and KV-caching
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

import sys
from pathlib import Path

# Add parent directory to path for imports when running from subdirectories
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .model_kvcache import GPTWithKVCache
    from .flash_attention_module import replace_attention_with_flash_attention
    from .model import GPTConfig, top_k_logits
except ImportError:
    # Fallback imports for when running from different directories
    from model_kvcache import GPTWithKVCache
    from flash_attention_module import replace_attention_with_flash_attention
    from model import GPTConfig, top_k_logits


class GPT_hf_KVCache(GPTWithKVCache):
    """
    A subclass of GPTWithKVCache that adds HuggingFace compatibility
    for use with TRL's GRPO trainer while maintaining KV-cache and Flash Attention.
    """
    
    def __init__(self, config):
        # Initialize with KV-cache and Flash Attention
        super().__init__(config, use_flash_attention=True)
        
        # Add HF compatibility attributes
        self.warnings_issued = {}
        self.is_peft_model = False
        self.hf = True
        
        # Beam search flags (for BGRPO)
        self.beam = False
        self.END_INDEX = None
        self.MASK_INDEX = None
    
    def add_model_tags(self, *args, **kwargs):
        """Dummy method for compatibility with GRPOTrainer."""
        pass  # Does nothing
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, 
                targets=None, return_attentions=False, use_cache=False, 
                output_hidden_states=False, **kwargs):
        """
        HuggingFace-compatible forward that uses KV-cache implementation.
        Maps HF keyword args to the KV-cache forward method.
        """
        # Map input_ids to idx for the parent method
        idx = input_ids
        
        # Always request hidden states from parent for TRL compatibility
        result = super().forward(
            idx=idx, 
            targets=targets, 
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_attentions=return_attentions,
            output_hidden_states=True  # Always get hidden states
        )
        
        # Parse the result based on what was returned
        if isinstance(result, tuple):
            if use_cache:
                if return_attentions:
                    # (logits, loss, cache, attentions, hidden_states)
                    logits, loss, past_key_values, attentions, hidden_states = result
                else:
                    # (logits, loss, cache, hidden_states)
                    logits, loss, past_key_values, hidden_states = result
                    attentions = None
            else:
                if return_attentions:
                    # (logits, loss, attentions, hidden_states)
                    logits, loss, attentions, hidden_states = result
                    past_key_values = None
                else:
                    # (logits, loss, hidden_states)
                    logits, loss, hidden_states = result
                    attentions = None
                    past_key_values = None
        else:
            raise ValueError(f"Unexpected return type from parent forward: {type(result)}")
        
        # Attentions are already extracted above
        
        # Return in HuggingFace format
        # We already have hidden_states from the parent forward call
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,  # From parent forward
            attentions=attentions,
        )
    
    @torch.no_grad()
    def beam_search_with_cache(self, x, steps, beam_width=3, temperature=1.0, 
                               pad_token=None, dataset=None):
        """
        Override beam search to handle HF-style forward outputs.
        Performs beam search with KV-cache optimization.
        """
        self.eval()
        device = x.device
        batch_size = x.size(0)
        
        # Initialize beams - start with single beam
        # Process initial sequence once
        output = self.forward(x, use_cache=True)
        initial_logits = output.logits
        initial_cache = output.past_key_values
        
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
                    new_cache = beam['cache']
                else:
                    # Get next token predictions using cache
                    last_token = beam['sequence'][:, -1:]
                    output = self.forward(
                        last_token, 
                        past_key_values=beam['cache'], 
                        use_cache=True
                    )
                    logits = output.logits
                    new_cache = output.past_key_values
                
                # Take last position logits and apply temperature
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Get top k tokens
                if step == 0:
                    # First expansion: get top beam_width tokens
                    topk_probs, topk_indices = torch.topk(probs[0], beam_width)
                    
                    for i in range(beam_width):
                        token = topk_indices[i].unsqueeze(0).unsqueeze(0)
                        score = torch.log(topk_probs[i]).item()
                        
                        new_sequence = torch.cat([beam['sequence'], token], dim=1)
                        
                        all_candidates.append({
                            'sequence': new_sequence,
                            'cache': new_cache,
                            'score': beam['score'] + score,
                            'finished': (pad_token is not None and token.item() == pad_token)
                        })
                else:
                    # Regular expansion: consider top beam_width tokens
                    topk_probs, topk_indices = torch.topk(probs[0], min(beam_width, probs.size(-1)))
                    
                    for i in range(topk_probs.size(0)):
                        token = topk_indices[i].unsqueeze(0).unsqueeze(0)
                        score = torch.log(topk_probs[i]).item()
                        
                        new_sequence = torch.cat([beam['sequence'], token], dim=1)
                        
                        all_candidates.append({
                            'sequence': new_sequence,
                            'cache': new_cache,
                            'score': beam['score'] + score,
                            'finished': (pad_token is not None and token.item() == pad_token)
                        })
            
            # Sort by score and keep top beam_width
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = all_candidates[:beam_width]
            
            # Check if all beams are finished
            if all(beam['finished'] for beam in beams):
                break
        
        # Extract sequences and pad them
        sequences = []
        for beam in beams:
            sequences.append(beam['sequence'].squeeze(0))  # Remove batch dimension
        
        # Pad sequences to same length
        if pad_token is not None:
            from torch.nn.utils.rnn import pad_sequence
            padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token)
        else:
            # Manual padding with zeros
            max_len = max(seq.size(0) for seq in sequences)
            padded_sequences = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype, device=device)
            for i, seq in enumerate(sequences):
                padded_sequences[i, :seq.size(0)] = seq
        
        return padded_sequences
    
    @torch.no_grad()
    def generate(self, input_ids, temperature=1.0, do_sample=True, top_k=None, 
                 eos_token_id=None, **kwargs):
        """
        Generate method that uses beam search when self.beam=True (for BGRPO).
        Otherwise uses KV-cache generation for efficiency.
        """
        self.eval()
        device = input_ids.device
        
        # Get generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 150)
        temperature = kwargs.get('temperature', temperature)
        do_sample = kwargs.get('do_sample', do_sample)
        top_k = kwargs.get('top_k', top_k)
        eos_token_id = kwargs.get('eos_token_id', getattr(self.config, 'eos_token_id', eos_token_id))
        pad_token_id = kwargs.get('pad_token_id', getattr(self.config, 'pad_token_id', None))
        
        if self.beam:
            # BGRPO mode: Use beam search with batch size as beam width
            print(f"[DEBUG generate] beam search enabled with KV-cache")
            print(f"[DEBUG generate] END_INDEX: {self.END_INDEX}, MASK_INDEX: {self.MASK_INDEX}")
            
            beam_width = len(input_ids)
            print(f"[DEBUG generate] input_ids shape: {input_ids.shape}")
            print(f"[DEBUG generate] beam width: {beam_width}")
            
            # Use beam search with KV-cache
            beam_sequences = self.beam_search_with_cache(
                input_ids[0:1],  # Just first prompt (they're all the same)
                max_new_tokens, 
                beam_width=beam_width,
                temperature=temperature,
                pad_token=pad_token_id,
                dataset=None  # Will be set by caller if needed
            )
            
            print(f"[DEBUG generate] beam result shape: {beam_sequences.shape}")
            return beam_sequences
        
        else:
            # Regular generation with KV-cache
            return self.generate_with_cache(
                input_ids, 
                max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k
            )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare inputs for generation step, handling KV-cache.
        """
        # If we have past_key_values, we only need to forward the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        # Add other potential inputs
        model_inputs.update({
            "attention_mask": kwargs.get("attention_mask"),
        })
        
        return model_inputs