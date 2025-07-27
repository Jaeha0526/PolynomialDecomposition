#!/usr/bin/env python3
"""
Test BGRPO training with KV-cache model
This tests if our model_hf_kvcache.py works correctly with BGRPO
"""

import os
import sys
import subprocess
import shutil

def modify_grpo_script_for_kvcache():
    """Modify grpo_ablation.py to use KV-cache"""
    
    # Read the original file
    with open("Training/BGRPO/grpo_ablation.py", "r") as f:
        content = f.read()
    
    # Check if it already has use_kvcache parameter
    if "use_kvcache" not in content:
        # Find the load_model_and_tokenizer call and add use_kvcache=True
        old_call = """model, tokenizer = load_model_and_tokenizer(
        config_path=str(config_path),
        model_dir_path=str(model_dir_path),
        device=device,
        wrap_for_grpo=True,
        model_name=args.model_name
    )"""
        
        new_call = """model, tokenizer = load_model_and_tokenizer(
        config_path=str(config_path),
        model_dir_path=str(model_dir_path),
        device=device,
        wrap_for_grpo=True,
        model_name=args.model_name,
        use_kvcache=True  # Enable KV-cache for faster beam search
    )"""
        
        if old_call in content:
            content = content.replace(old_call, new_call)
            print("✅ Modified grpo_ablation.py to use KV-cache")
            return content
        else:
            print("⚠️  Could not find exact match, trying alternative approach")
            # Try a more flexible replacement
            import re
            pattern = r'(model, tokenizer = load_model_and_tokenizer\([^)]+)\)'
            match = re.search(pattern, content)
            if match:
                old_text = match.group(0)
                new_text = old_text.rstrip(')') + ',\n        use_kvcache=True  # Enable KV-cache for faster beam search\n    )'
                content = content.replace(old_text, new_text)
                print("✅ Modified grpo_ablation.py to use KV-cache (alternative method)")
                return content
    else:
        print("ℹ️  grpo_ablation.py already has use_kvcache parameter")
    
    return None

def run_quick_bgrpo_test():
    """Run a quick BGRPO test with KV-cache"""
    
    os.chdir('/workspace/PolynomialDecomposition')
    
    # Backup original file
    shutil.copy("Training/BGRPO/grpo_ablation.py", "Training/BGRPO/grpo_ablation.py.backup")
    
    # Modify the script
    modified_content = modify_grpo_script_for_kvcache()
    
    if modified_content:
        with open("Training/BGRPO/grpo_ablation.py", "w") as f:
            f.write(modified_content)
    
    # Create a quick test script
    test_script = """#!/bin/bash
# Quick BGRPO test with KV-cache

export CUDA_VISIBLE_DEVICES=0

MODEL_FILE="single_variable_model_best.pt"
CONFIG_FILE="model_configuration.json"

echo "Testing BGRPO with KV-cache model"
python grpo_ablation.py \
  --model_name ${MODEL_FILE} \
  --reward_type rank \
  --output_dir ../../test/bgrpo_kvcache_test \
  --config_name ${CONFIG_FILE} \
  --dataset_path ../../data_storage/dataset/single_variable/training_dataset.txt \
  --disable_wandb \
  --lr 1e-5 \
  --beta 0.01 \
  --total_training_samples 20 \
  --num_generations 5 \
  --num_questions 4 \
  --num_iterations 2 \
  --save_steps 1

echo "BGRPO KV-cache test completed!"
"""
    
    with open("test/test_bgrpo_kvcache.sh", "w") as f:
        f.write(test_script)
    
    os.chmod("test/test_bgrpo_kvcache.sh", 0o755)
    
    print("\n" + "="*80)
    print("BGRPO KV-CACHE TEST")
    print("="*80)
    print("\nRunning a quick BGRPO test with KV-cache enabled...")
    print("This will:")
    print("1. Load model with GPT_hf_KVCache")
    print("2. Run beam search with KV-cache optimization")
    print("3. Train for 2 iterations with 4 questions per iteration")
    print("\n" + "="*80)
    
    # Run the test
    try:
        result = subprocess.run(
            ["bash", "test/test_bgrpo_kvcache.sh"],
            cwd="Training/BGRPO",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("\n📊 RESULTS:")
        print("-" * 80)
        
        # Check for successful model loading
        if "GPT_hf_KVCache model instantiated" in result.stdout:
            print("✅ KV-cache model loaded successfully")
        else:
            print("❌ KV-cache model loading not confirmed")
        
        # Check for beam search
        if "beam search enabled" in result.stdout:
            print("✅ Beam search is running")
        
        # Check for training progress
        if "loss" in result.stdout:
            print("✅ Training is progressing")
            # Extract some loss values
            import re
            losses = re.findall(r"'loss': ([\d.]+)", result.stdout)
            if losses:
                print(f"   Sample losses: {losses[:3]}")
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n⚠️  Process exited with code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])
        
        # Show output
        print("\n📝 Output excerpt:")
        print("-" * 80)
        print(result.stdout[-1000:])  # Last 1000 chars
        
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")
    
    finally:
        # Restore original file
        shutil.move("Training/BGRPO/grpo_ablation.py.backup", "Training/BGRPO/grpo_ablation.py")
        print("\n✅ Restored original grpo_ablation.py")

def check_model_compatibility():
    """Quick check to ensure model_hf_kvcache.py is compatible"""
    
    print("\n" + "="*80)
    print("CHECKING MODEL COMPATIBILITY")
    print("="*80)
    
    try:
        sys.path.append('/workspace/PolynomialDecomposition')
        from Training.mingpt.model_hf_kvcache import GPT_hf_KVCache
        from Training.mingpt.model import GPTConfig
        
        # Create a small test model
        config = GPTConfig(vocab_size=50, block_size=128, n_layer=2, n_head=2, n_embd=64)
        model = GPT_hf_KVCache(config)
        
        print("✅ model_hf_kvcache.py imports successfully")
        print(f"✅ Created test model: {type(model).__name__}")
        
        # Check required attributes
        required_attrs = ['beam', 'END_INDEX', 'MASK_INDEX', 'forward', 'beam_search_with_cache']
        for attr in required_attrs:
            if hasattr(model, attr):
                print(f"✅ Has required attribute: {attr}")
            else:
                print(f"❌ Missing required attribute: {attr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False

if __name__ == "__main__":
    # First check compatibility
    if check_model_compatibility():
        # Run the BGRPO test
        run_quick_bgrpo_test()
    else:
        print("\n⚠️  Skipping BGRPO test due to compatibility issues")