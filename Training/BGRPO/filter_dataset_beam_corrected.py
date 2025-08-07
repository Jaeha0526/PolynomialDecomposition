#!/usr/bin/env python3
"""
Filter training dataset to keep only examples where the correct answer 
appears in the top-k beams during beam search.

This ensures positive rewards during BGRPO training.
"""

import torch
import json
import time
from types import SimpleNamespace
from tqdm import tqdm
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mingpt.model import GPT_hf
from mingpt.utils import LLM_BeamSearch_check, is_valid_expression_sympy_single
from mingpt.model_loader import load_model_and_tokenizer

def check_beam_search_solvable(model, tokenizer, prompt, target, beam_width=25, device="cuda"):
    """Check if the correct answer appears in top-k beams"""
    model.eval()
    
    # Get the base model if it's wrapped
    base_model = model.pretrained_model if hasattr(model, 'pretrained_model') else model
    
    # Set beam search parameters
    base_model.beam = True
    base_model.END_INDEX = tokenizer.eos_token_id
    base_model.MASK_INDEX = tokenizer.mask_token_id
    
    # Prepare args for beam search
    args_beam = SimpleNamespace(
        beam_width=beam_width,
        max_output_length=150,
        check_path=None,
        hf=True,
        sympy=1
    )
    
    # Get input string (everything before MASK_CHAR)
    input_str = prompt.split(tokenizer.MASK_CHAR)[0]
    
    try:
        with torch.no_grad():
            pred_str, correct_beam_rank = LLM_BeamSearch_check(
                base_model, input_str, tokenizer, device, args_beam
            )
        
        # Debug first few examples
        global debug_count
        if debug_count < 3:
            print(f"\nDEBUG Example {debug_count + 1}:")
            print(f"  Input: {input_str[:50]}...")
            print(f"  Target: {target}")
            print(f"  Prediction: {pred_str}")
            print(f"  Beam rank: {correct_beam_rank}")
            debug_count += 1
        
        # correct_beam_rank is 1-indexed, -1 means not found
        if correct_beam_rank > 0 and correct_beam_rank <= beam_width:
            return True, correct_beam_rank
        else:
            return False, -1
            
    except Exception as e:
        print(f"Error during beam search: {e}")
        return False, -1

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by beam search solvability")
    parser.add_argument("--model", type=str, 
                        default="/workspace/PolynomialDecomposition/data_storage/model/",
                        help="Model directory path")
    parser.add_argument("--config", type=str, 
                        default="/workspace/PolynomialDecomposition/data_storage/model/model_configurations/model_configuration.json",
                        help="Model configuration file")
    parser.add_argument("--input_dataset", type=str, 
                        default="../../data_storage/dataset/single_variable/training_dataset_4_4_filtered.txt",
                        help="Input dataset path")
    parser.add_argument("--output_dataset", type=str,
                        default="../../data_storage/dataset/single_variable/training_dataset_4_4_beam25_500samples.txt",
                        help="Output dataset path")
    parser.add_argument("--beam_width", type=int, default=25,
                        help="Beam width to use for filtering")
    parser.add_argument("--target_samples", type=int, default=500,
                        help="Target number of filtered samples to collect")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Process in batches for progress tracking")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.config,  # config file path first
        args.model,   # model directory path second
        use_kvcache=False
    )
    model = model.to(device)
    model.eval()
    
    # Set up for beam search
    base_model = model.pretrained_model if hasattr(model, 'pretrained_model') else model
    base_model.beam = True
    base_model.END_INDEX = tokenizer.eos_token_id
    base_model.MASK_INDEX = tokenizer.mask_token_id
    base_model.hf = True  # Ensure hf mode is set
    
    print(f"Model loaded. Beam search enabled.")
    print(f"END_INDEX: {base_model.END_INDEX}, MASK_INDEX: {base_model.MASK_INDEX}")
    
    # Read input dataset
    print(f"\nReading dataset from {args.input_dataset}...")
    with open(args.input_dataset, 'r') as f:
        lines = f.readlines()
    
    print(f"Total examples in dataset: {len(lines)}")
    print(f"Target: Find {args.target_samples} examples solvable with beam_width={args.beam_width}")
    
    # Process dataset
    filtered_lines = []
    beam_rank_distribution = {}
    
    print(f"\nFiltering dataset...")
    
    start_time = time.time()
    processed = 0
    
    # Global debug counter
    global debug_count
    debug_count = 0
    
    # Process in batches with progress bar
    with tqdm(total=args.target_samples, desc="Finding solvable examples") as pbar:
        for i in range(0, len(lines), args.batch_size):
            if len(filtered_lines) >= args.target_samples:
                break
                
            batch_lines = lines[i:i+args.batch_size]
            
            for line in batch_lines:
                if len(filtered_lines) >= args.target_samples:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # Parse the line - format is "prompt⁇target" (no TAB)
                if tokenizer.MASK_CHAR not in line:
                    print(f"Warning: Skipping line without MASK_CHAR: {line[:50]}...")
                    continue
                
                # Split by MASK_CHAR - the prompt includes the MASK_CHAR at the end
                mask_idx = line.find(tokenizer.MASK_CHAR)
                prompt = line[:mask_idx + 1]  # Include MASK_CHAR
                target = line[mask_idx + 1:].strip()  # Everything after MASK_CHAR
                
                processed += 1
                
                # Check if solvable with beam search
                is_solvable, beam_rank = check_beam_search_solvable(
                    model, tokenizer, prompt, target, args.beam_width, device
                )
                
                if is_solvable:
                    filtered_lines.append(line)
                    # Track distribution of beam ranks
                    if beam_rank not in beam_rank_distribution:
                        beam_rank_distribution[beam_rank] = 0
                    beam_rank_distribution[beam_rank] += 1
                    pbar.update(1)
                
                # Print progress stats periodically
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    found_rate = len(filtered_lines) / processed if processed > 0 else 0
                    
                    tqdm.write(f"Processed: {processed}, Found: {len(filtered_lines)} ({found_rate*100:.1f}%), Rate: {rate:.1f} ex/sec")
    
    # Save filtered dataset
    print(f"\n\nSaving filtered dataset to {args.output_dataset}...")
    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)
    
    with open(args.output_dataset, 'w') as f:
        for line in filtered_lines[:args.target_samples]:  # Ensure we don't exceed target
            f.write(line + '\n')
    
    # Print statistics
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"Total examples processed: {processed}")
    print(f"Examples found: {len(filtered_lines)}")
    print(f"Final dataset size: {min(len(filtered_lines), args.target_samples)}")
    print(f"Success rate: {100*len(filtered_lines)/processed:.1f}%")
    print(f"Time taken: {time.time() - start_time:.1f} seconds")
    
    if len(beam_rank_distribution) > 0:
        print(f"\nBeam rank distribution:")
        for rank in sorted(beam_rank_distribution.keys())[:10]:  # Show top 10
            count = beam_rank_distribution[rank]
            percentage = 100 * count / len(filtered_lines)
            print(f"  Rank {rank}: {count} examples ({percentage:.1f}%)")
        
        # Calculate cumulative distribution
        print(f"\nCumulative distribution:")
        cumulative = 0
        for rank in sorted(beam_rank_distribution.keys())[:5]:  # Show top 5
            cumulative += beam_rank_distribution[rank]
            percentage = 100 * cumulative / len(filtered_lines)
            print(f"  Top-{rank}: {cumulative} examples ({percentage:.1f}%)")
    
    print(f"\nFiltered dataset saved to: {args.output_dataset}")

if __name__ == "__main__":
    main()