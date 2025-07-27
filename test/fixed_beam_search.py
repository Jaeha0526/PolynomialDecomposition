#!/usr/bin/env python3
"""
Fixed version of beam_search_with_cache that properly handles variable-length sequences
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append('/workspace/PolynomialDecomposition')
sys.path.append('/workspace/PolynomialDecomposition/Training/mingpt')
os.chdir('/workspace/PolynomialDecomposition')

from Training.mingpt.model_kvcache import GPTWithKVCache
from Training.mingpt import model

def beam_search_with_cache_fixed(self, x, max_new_tokens, beam_width=3, temperature=1.0, 
                                  pad_token=None, dataset=None):
    """
    Fixed beam search that properly handles variable-length sequences by padding
    """
    # Ensure model is in eval mode
    self.eval()
    
    # Initial forward pass to get logits and cache
    initial_logits, _, initial_cache = self.forward(x, use_cache=True)
    
    # Initialize beams with the input sequence
    beams = [{
        'sequence': x,
        'cache': initial_cache,
        'score': 0.0,
        'finished': False
    }]
    
    for step in range(max_new_tokens):
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


def test_fixed_beam_search():
    """Test the fixed beam search implementation"""
    from Training.mingpt import dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    chars_symbolic = [
        "□",
        "a","b","c","d","e","x","y","z",
        "⁇","?",
        "a0","a1","b0","b1",
        "N","P","&","+","*","^",
    ] + [str(i) for i in range(0, 10)]
    
    test_dataset = dataset.SymbolicDataset(
        300,
        chars_symbolic,
        open("data_storage/dataset/single_variable/test_dataset_2_2.txt", encoding="utf-8").read(),
    )
    
    # Create KV-cache model
    model_cfg = model.GPTConfig(
        vocab_size=len(chars_symbolic), 
        block_size=300, 
        n_layer=6, 
        n_head=8, 
        n_embd=512
    )
    gpt = GPTWithKVCache(model_cfg, use_flash_attention=True)
    gpt.to(device)
    
    # Monkey-patch the fixed method
    gpt.beam_search_with_cache = lambda *args, **kwargs: beam_search_with_cache_fixed(gpt, *args, **kwargs)
    
    # Load weights
    print("Loading model weights...")
    gpt.load_state_dict(torch.load("data_storage/model/single_variable_model_best.pt"), strict=False)
    
    print(f"Model type: {type(gpt).__name__}")
    print("✅ Using FIXED beam_search_with_cache method")
    
    # Test multiple inputs to see if variable lengths cause issues
    test_lines = [
        "+ N 2 4 0 + * P 2 1 5 a + * P 6 3 8 ^ a P 2 + * N 3 2 0 ^ a P 3 * N 5 1 2 ^ a P 4 ⁇",
        "+ P 1 0 8 + * N 5 5 2 a + * N 5 1 2 ^ a P 2 + * P 2 9 6 4 ^ a P 3 * P 3 2 1 1 ^ a P 4 ⁇",
        "+ N 1 0 2 4 + * P 2 0 5 2 a + * N 2 8 8 9 ^ a P 2 + * P 1 8 4 8 ^ a P 3 * N 8 4 7 ^ a P 4 ⁇",
    ]
    
    pad_token = test_dataset.stoi[test_dataset.PAD_CHAR] if hasattr(test_dataset, 'PAD_CHAR') else None
    
    success_count = 0
    for i, test_line in enumerate(test_lines):
        print(f"\n{'='*60}")
        print(f"Test {i+1}:")
        
        # Prepare input
        line_here = test_line.replace("?", "⁇")
        x = line_here.split("⁇")[0]
        x = x.split(" ")
        x.append("⁇")
        x = [item for item in x if item != ""]
        x_tensor = torch.tensor([test_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
        
        print(f"Input shape: {x_tensor.shape}")
        
        # Test beam search
        try:
            beam_sequences = gpt.beam_search_with_cache(
                x_tensor, 150, beam_width=3, temperature=1.0, 
                pad_token=pad_token, dataset=test_dataset
            )
            
            print("✅ Beam search completed successfully!")
            print(f"Output shape: {beam_sequences.shape}")
            
            # Show results for each beam
            for j in range(min(3, beam_sequences.size(0))):
                seq = beam_sequences[j]
                # Convert to string (stop at END token if present)
                tokens = []
                for token_id in seq:
                    if hasattr(test_dataset, 'END_INDEX') and token_id == test_dataset.END_INDEX:
                        break
                    if token_id != pad_token and token_id < len(test_dataset.itos):
                        tokens.append(test_dataset.itos[int(token_id)])
                
                output_str = " ".join(tokens)
                print(f"\nBeam {j}: {output_str[:80]}...")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Beam search failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(test_lines)} tests passed")
    
    if success_count == len(test_lines):
        print("\n🎉 All tests passed! The fixed beam search handles variable-length sequences correctly.")
        print("\nKey improvements:")
        print("1. Sequences are padded to the same length before stacking")
        print("2. Each beam maintains its own KV-cache (already implemented)")
        print("3. Variable-length outputs no longer cause tensor size mismatches")

if __name__ == "__main__":
    test_fixed_beam_search()