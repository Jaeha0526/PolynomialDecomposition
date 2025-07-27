#!/usr/bin/env python3
"""
Direct test of BGRPO with KV-cache
"""

import os
import sys
import subprocess
import time

def test_bgrpo_kvcache():
    """Test BGRPO with KV-cache directly"""
    
    os.chdir('/workspace/PolynomialDecomposition')
    
    # First, check if the model file exists
    model_path = "data_storage/model/single_variable_model_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("="*80)
    print("TESTING BGRPO WITH KV-CACHE")
    print("="*80)
    
    # Temporarily modify model_loader.py to see debug output
    cmd = [
        "python", "Training/BGRPO/grpo_ablation.py",
        "--model_name", "single_variable_model_best.pt",
        "--reward_type", "rank",
        "--output_dir", "test/bgrpo_kvcache_test_output",
        "--config_name", "model_configuration.json",
        "--dataset_path", "data_storage/dataset/single_variable/training_dataset.txt",
        "--disable_wandb",
        "--lr", "1e-5",
        "--beta", "0.01",
        "--total_training_samples", "16",  # Very small for quick test
        "--num_generations", "4",
        "--num_questions", "2",
        "--num_iterations", "2",
        "--save_steps", "1"
    ]
    
    # Add use_kvcache to model_loader temporarily
    print("📝 Modifying model_loader.py to use KV-cache...")
    
    with open("Training/mingpt/model_loader.py", "r") as f:
        content = f.read()
    
    # Backup
    with open("Training/mingpt/model_loader.py.backup", "w") as f:
        f.write(content)
    
    # Find and modify the load_model_and_tokenizer call
    old_line = "    model, tokenizer = load_model_and_tokenizer("
    new_content = content.replace(
        "        wrap_for_grpo=True,\n        model_name=args.model_name",
        "        wrap_for_grpo=True,\n        model_name=args.model_name,\n        use_kvcache=True"
    )
    
    if new_content != content:
        with open("Training/mingpt/model_loader.py", "w") as f:
            f.write(new_content)
        print("✅ Modified model_loader.py")
    
    # Also modify grpo_ablation.py
    with open("Training/BGRPO/grpo_ablation.py", "r") as f:
        grpo_content = f.read()
    
    # Backup
    with open("Training/BGRPO/grpo_ablation.py.backup", "w") as f:
        f.write(grpo_content)
    
    # Modify the load call
    new_grpo = grpo_content.replace(
        "        wrap_for_grpo=False,\n        model_name=args.model_name",
        "        wrap_for_grpo=True,\n        model_name=args.model_name,\n        use_kvcache=True"
    )
    
    if new_grpo != grpo_content:
        with open("Training/BGRPO/grpo_ablation.py", "w") as f:
            f.write(new_grpo)
        print("✅ Modified grpo_ablation.py")
    
    print("\n🚀 Running BGRPO test with KV-cache...")
    print("Command:", " ".join(cmd))
    print("-"*80)
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        kvcache_loaded = False
        training_started = False
        beam_search_working = False
        
        # Read output line by line
        for line in process.stdout:
            output_lines.append(line)
            print(line.rstrip())
            
            # Check for key indicators
            if "GPT_hf_KVCache model instantiated" in line:
                kvcache_loaded = True
            if "loss" in line and "epoch" in line:
                training_started = True
            if "beam search enabled" in line or "Beam" in line:
                beam_search_working = True
            
            # Stop after some progress
            if len(output_lines) > 100 and training_started:
                print("\n... (stopping early test) ...")
                process.terminate()
                break
        
        process.wait()
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        print(f"\n⏱️  Test ran for {elapsed:.1f} seconds")
        
        if kvcache_loaded:
            print("✅ KV-cache model loaded successfully!")
        else:
            print("❌ KV-cache model was not loaded")
        
        if beam_search_working:
            print("✅ Beam search is working")
        else:
            print("⚠️  Beam search not confirmed")
        
        if training_started:
            print("✅ BGRPO training started successfully")
        else:
            print("❌ Training did not start")
        
        # Show relevant output
        print("\n📝 Key output lines:")
        for line in output_lines:
            if any(key in line for key in ["KVCache", "Flash", "beam", "loss", "Error", "error"]):
                print(f"  {line.rstrip()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        # Restore original files
        print("\n🔄 Restoring original files...")
        if os.path.exists("Training/mingpt/model_loader.py.backup"):
            os.rename("Training/mingpt/model_loader.py.backup", "Training/mingpt/model_loader.py")
        if os.path.exists("Training/BGRPO/grpo_ablation.py.backup"):
            os.rename("Training/BGRPO/grpo_ablation.py.backup", "Training/BGRPO/grpo_ablation.py")
        print("✅ Files restored")

if __name__ == "__main__":
    test_bgrpo_kvcache()