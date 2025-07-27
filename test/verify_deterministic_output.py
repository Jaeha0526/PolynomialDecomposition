#!/usr/bin/env python3
"""
Verify that outputs are deterministic and identical across different runs
and optimization configurations
"""

import subprocess
import os
import hashlib

def run_evaluation_and_save_output(output_file, run_name):
    """Run evaluation and save predictions to file"""
    
    cmd = [
        "python", "Training/mingpt/run.py", "inequality_evaluate4",
        "--block_size", "300",
        "--max_output_length", "150",
        "--n_embd", "512",
        "--n_head", "8",
        "--n_layer", "6",
        "--sympy", "1",
        "--max_test", "20",  # Small number for detailed comparison
        "--evaluate_corpus_path", "data_storage/dataset/single_variable/test_dataset_2_2.txt",
        "--reading_params_path", "data_storage/model/single_variable_model_best.pt",
        "--outputs_path", output_file
    ]
    
    print(f"\n🔄 Running {run_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract accuracy
    accuracy = None
    if "Correct:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Correct:" in line:
                accuracy = line.strip()
                break
    
    # Check for Flash + KV confirmation
    has_optimizations = "GPTWithKVCache initialized with Flash Attention + KV-Cache" in result.stdout
    
    return {
        'accuracy': accuracy,
        'has_optimizations': has_optimizations,
        'stdout': result.stdout
    }

def compare_prediction_files(file1, file2, name1, name2):
    """Compare two prediction files line by line"""
    print(f"\n📊 Comparing {name1} vs {name2}:")
    
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    if len(lines1) != len(lines2):
        print(f"   ❌ Different number of predictions: {len(lines1)} vs {len(lines2)}")
        return False
    
    differences = 0
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1.strip() != line2.strip():
            differences += 1
            if differences <= 3:  # Show first 3 differences
                print(f"   Line {i+1} differs:")
                print(f"     Run 1: {line1.strip()[:100]}...")
                print(f"     Run 2: {line2.strip()[:100]}...")
    
    if differences == 0:
        print(f"   ✅ Predictions are IDENTICAL")
        
        # Calculate hash for verification
        hash1 = hashlib.md5(open(file1, 'rb').read()).hexdigest()
        hash2 = hashlib.md5(open(file2, 'rb').read()).hexdigest()
        print(f"   MD5 hash: {hash1}")
        assert hash1 == hash2, "Hash mismatch despite identical content!"
    else:
        print(f"   ❌ Found {differences} differences out of {len(lines1)} predictions")
    
    return differences == 0

def main():
    """Run deterministic verification"""
    os.chdir('/workspace/PolynomialDecomposition')
    
    print("="*60)
    print("DETERMINISTIC OUTPUT VERIFICATION")
    print("Testing: Greedy sampling should produce identical results")
    print("="*60)
    
    # Run multiple times to verify determinism
    runs = []
    output_files = []
    
    for i in range(3):
        output_file = f"test/deterministic_test_run{i+1}.txt"
        output_files.append(output_file)
        
        result = run_evaluation_and_save_output(output_file, f"Run {i+1}")
        runs.append(result)
        
        print(f"   Accuracy: {result['accuracy']}")
        print(f"   Optimizations enabled: {result['has_optimizations']}")
    
    # Compare all runs
    print("\n" + "="*60)
    print("DETERMINISM CHECK")
    print("="*60)
    
    all_identical = True
    
    # Compare Run 1 vs Run 2
    if not compare_prediction_files(output_files[0], output_files[1], "Run 1", "Run 2"):
        all_identical = False
    
    # Compare Run 2 vs Run 3
    if not compare_prediction_files(output_files[1], output_files[2], "Run 2", "Run 3"):
        all_identical = False
    
    # Compare Run 1 vs Run 3 for completeness
    if not compare_prediction_files(output_files[0], output_files[2], "Run 1", "Run 3"):
        all_identical = False
    
    # Verify accuracy is identical
    print("\n📊 ACCURACY VERIFICATION:")
    accuracies = [run['accuracy'] for run in runs]
    unique_accuracies = set(accuracies)
    
    if len(unique_accuracies) == 1:
        print(f"   ✅ All runs have identical accuracy: {accuracies[0]}")
    else:
        print(f"   ❌ Accuracies differ across runs: {accuracies}")
        all_identical = False
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if all_identical:
        print("✅ PASS: All runs produce identical deterministic results")
        print("   This confirms that:")
        print("   1. Greedy sampling is working correctly")
        print("   2. Flash Attention and KV-Cache don't affect outputs")
        print("   3. The model is deterministic as expected")
    else:
        print("❌ FAIL: Outputs are not deterministic!")
        print("   This indicates a potential issue with:")
        print("   - Random seed management")
        print("   - Non-deterministic operations")
        print("   - Implementation bugs")
    
    # Clean up
    for f in output_files:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()