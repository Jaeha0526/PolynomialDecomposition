#!/usr/bin/env python3
"""
Verify that beam search produces identical outputs across all optimization configurations
"""

import torch
import os
import sys
import hashlib
sys.path.append('/workspace/PolynomialDecomposition')
sys.path.append('/workspace/PolynomialDecomposition/Training/mingpt')

from Training.mingpt import model, dataset, utils
from Training.mingpt.model_kvcache import GPTWithKVCache
from Training.mingpt.flash_attention_module import replace_attention_with_flash_attention

def get_beam_search_output(gpt, x_tensor, test_dataset, beam_width=5):
    """Get beam search output for a given model and input"""
    
    if hasattr(gpt, 'beam_search_with_cache'):
        # Use KV-cache beam search
        pad_token = test_dataset.stoi[test_dataset.PAD_CHAR] if hasattr(test_dataset, 'PAD_CHAR') else None
        beam_sequences = gpt.beam_search_with_cache(
            x_tensor, 150, beam_width=beam_width, 
            temperature=1.0, pad_token=pad_token, dataset=test_dataset
        )
        
        # Convert to string format
        beam_results = []
        for i in range(beam_sequences.size(0)):
            seq = beam_sequences[i]
            # Convert to tokens, stopping at padding
            tokens = []
            for token_id in seq[x_tensor.size(1):]:  # Skip input tokens
                if hasattr(test_dataset, 'END_INDEX') and token_id == test_dataset.END_INDEX:
                    break
                if pad_token is not None and token_id == pad_token:
                    break
                if token_id < len(test_dataset.itos):
                    tokens.append(test_dataset.itos[int(token_id)])
            beam_str = " ".join(tokens)
            beam_results.append(beam_str)
        return beam_results
    else:
        # Use original beam search
        beam_result = utils.beam_search(
            gpt, x_tensor, 150, test_dataset, 
            beam_width=beam_width, temperature=1.0, hf=False
        )
        
        # Extract just the generated part (after ⁇)
        processed_results = []
        for beam_str, _, _ in beam_result:
            # Split by ⁇ and take the part after it
            if "⁇" in beam_str:
                generated = beam_str.split("⁇")[1].strip()
            else:
                generated = beam_str
            processed_results.append(generated)
        return processed_results

def main():
    """Verify outputs match across configurations"""
    os.chdir('/workspace/PolynomialDecomposition')
    
    print("="*80)
    print("BEAM SEARCH OUTPUT VERIFICATION")
    print("Checking if all optimization configurations produce identical results")
    print("="*80)
    
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
    
    # Model config
    model_cfg = model.GPTConfig(
        vocab_size=len(chars_symbolic), 
        block_size=300, 
        n_layer=6, 
        n_head=8, 
        n_embd=512
    )
    
    # Test samples
    test_lines = [
        "+ N 2 4 0 + * P 2 1 5 a + * P 6 3 8 ^ a P 2 + * N 3 2 0 ^ a P 3 * N 5 1 2 ^ a P 4 ⁇",
        "+ P 1 0 8 + * N 5 5 2 a + * N 5 1 2 ^ a P 2 + * P 2 9 6 4 ^ a P 3 * P 3 2 1 1 ^ a P 4 ⁇",
        "+ N 1 0 2 4 + * P 2 0 5 2 a + * N 2 8 8 9 ^ a P 2 + * P 1 8 4 8 ^ a P 3 * N 8 4 7 ^ a P 4 ⁇",
    ]
    
    # Configurations to test
    configs = [
        ("No optimizations", lambda: model.GPT(model_cfg)),
        ("Flash Attention only", lambda: replace_attention_with_flash_attention(model.GPT(model_cfg))),
        ("Flash + KV-cache", lambda: GPTWithKVCache(model_cfg, use_flash_attention=True)),
    ]
    
    beam_width = 5
    
    # Store results for each configuration
    all_results = {}
    
    for config_name, model_factory in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print('='*60)
        
        # Create and load model
        gpt = model_factory()
        gpt.to(device)
        gpt.load_state_dict(torch.load("data_storage/model/single_variable_model_best.pt"), strict=False)
        gpt.eval()
        
        print(f"Model type: {type(gpt).__name__}")
        
        results = []
        
        for i, test_line in enumerate(test_lines):
            # Prepare input
            line_here = test_line.replace("?", "⁇")
            x = line_here.split("⁇")[0]
            x = x.split(" ")
            x.append("⁇")
            x = [item for item in x if item != ""]
            x_tensor = torch.tensor([test_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
            
            # Get beam search results
            beam_results = get_beam_search_output(gpt, x_tensor, test_dataset, beam_width)
            
            results.append(beam_results)
            
            print(f"\nSample {i+1} - Top beam result:")
            print(f"  {beam_results[0][:80]}..." if len(beam_results[0]) > 80 else f"  {beam_results[0]}")
        
        all_results[config_name] = results
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print('='*80)
    
    configs_list = list(all_results.keys())
    baseline = configs_list[0]
    all_match = True
    
    for sample_idx in range(len(test_lines)):
        print(f"\n📝 Sample {sample_idx + 1}:")
        
        # Compare each beam position
        for beam_idx in range(beam_width):
            baseline_result = all_results[baseline][sample_idx][beam_idx]
            
            matches = []
            for config in configs_list[1:]:
                config_result = all_results[config][sample_idx][beam_idx]
                match = baseline_result == config_result
                matches.append((config, match))
                
                if not match:
                    all_match = False
            
            if all([m[1] for m in matches]):
                if beam_idx == 0:  # Only show for top beam
                    print(f"  ✅ Beam {beam_idx + 1}: All configurations match")
            else:
                print(f"  ❌ Beam {beam_idx + 1}: Mismatch detected!")
                print(f"     {baseline}: {baseline_result[:50]}...")
                for config, match in matches:
                    if not match:
                        print(f"     {config}: {all_results[config][sample_idx][beam_idx][:50]}...")
    
    # Calculate hashes for verification
    print(f"\n{'='*60}")
    print("HASH VERIFICATION")
    print('='*60)
    
    for config_name, results in all_results.items():
        # Create a string of all results
        all_text = ""
        for sample_results in results:
            for beam_result in sample_results:
                all_text += beam_result + "\n"
        
        # Calculate hash
        hash_value = hashlib.md5(all_text.encode()).hexdigest()
        print(f"{config_name:<25}: {hash_value}")
    
    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print('='*80)
    
    if all_match:
        print("✅ SUCCESS: All optimization configurations produce IDENTICAL beam search results!")
        print("   This confirms that:")
        print("   1. The optimizations don't affect model outputs")
        print("   2. Flash Attention is mathematically equivalent")
        print("   3. KV-cache correctly maintains state")
        print("   4. The beam search fix works correctly")
    else:
        print("❌ FAILURE: Outputs differ between configurations!")
        print("   This indicates a potential bug in one of the implementations")
    
    print("\n💡 Key Takeaway:")
    print("   Performance optimizations should never change model outputs.")
    print("   If they do, there's likely a bug in the implementation.")

if __name__ == "__main__":
    main()