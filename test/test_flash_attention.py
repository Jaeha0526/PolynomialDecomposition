#!/usr/bin/python3
"""
Test script to benchmark Flash Attention model and compare with baseline
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

# Add path to mingpt
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))

# Import both models
from model import GPT, GPTConfig
from model_flash_attention import FlashGPT
import dataset

# Set device - try GPU but fallback to CPU
try:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    test_tensor = torch.randn(1, 1).to(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    device = torch.device("cpu")
    print(f"Using device: {device} (GPU init failed)")

# Configuration
BLOCK_SIZE = 400
N_EMBD = 256
N_LAYER = 4
N_HEAD = 8
BATCH_SIZE = 32 if device.type == "cpu" else 256
MAX_ITER = 20  # Number of iterations to benchmark

# Define vocabulary
chars_symbolic = [
    "□",
    "a","b","c","d","e","x","y","z",
    "⁇","?",
    "a0","a1","b0","b1",
    "N","P","&","+","*","^",
] + [str(i) for i in range(0, 10)]

# Extend for ON dataset
for i in range(2, 21):
    chars_symbolic.extend([f"a{i}", f"b{i}"])

vocab_size = len(chars_symbolic)
print(f"Vocabulary size: {vocab_size}")
print(f"Batch size: {BATCH_SIZE}")

# Load dataset
print("\nLoading ON validation dataset...")
valid_data_path = "data_storage/dataset/example_ON/ON_data_valid.txt"
with open(valid_data_path, 'r') as f:
    valid_data = f.read()

valid_dataset = dataset.SymbolicDataset(BLOCK_SIZE, chars_symbolic, valid_data)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Valid dataset size: {len(valid_dataset):,}")

# Create model configuration
model_config = GPTConfig(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD
)

# Create both models
print("\nCreating models...")
baseline_model = GPT(model_config).to(device)
flash_model = FlashGPT(model_config).to(device)

# Ensure same initialization
flash_model.load_state_dict(baseline_model.state_dict(), strict=False)

def benchmark_model(model, loader, model_name, num_iters=20):
    """Benchmark a model's forward pass"""
    model.eval()
    times = []
    
    print(f"\nBenchmarking {model_name}...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_iters:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Warmup
            if i == 0:
                print(f"Warming up {model_name}...")
                for _ in range(3):
                    _ = model(x, targets=y)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            
            # Time the forward pass
            start = time.time()
            logits, loss = model(x, targets=y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
            
            if i == 0:
                print(f"First batch loss: {loss.item():.4f}")
    
    # Calculate statistics (exclude warmup)
    times = times[1:] if len(times) > 1 else times
    avg_time = sum(times) / len(times) if times else 0
    throughput = BATCH_SIZE / avg_time if avg_time > 0 else 0
    
    return {
        "model": model_name,
        "avg_time_ms": avg_time * 1000,
        "throughput_samples_per_sec": throughput,
        "min_time_ms": min(times) * 1000 if times else 0,
        "max_time_ms": max(times) * 1000 if times else 0,
        "num_iterations": len(times),
        "device": str(device),
        "batch_size": BATCH_SIZE,
    }

# Run benchmarks
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

baseline_results = benchmark_model(baseline_model, valid_loader, "Baseline GPT", MAX_ITER)
flash_results = benchmark_model(flash_model, valid_loader, "Flash Attention GPT", MAX_ITER)

# Memory usage
if device.type == "cuda":
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Run one forward pass to measure memory
    x, y = next(iter(valid_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        _ = flash_model(x, targets=y)
    
    flash_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    baseline_results["peak_memory_mb"] = baseline_memory
    flash_results["peak_memory_mb"] = flash_memory

# Calculate improvements
speedup = baseline_results["avg_time_ms"] / flash_results["avg_time_ms"]
throughput_improvement = flash_results["throughput_samples_per_sec"] / baseline_results["throughput_samples_per_sec"]

# Print results
print("\nRESULTS SUMMARY")
print("-" * 60)
print(f"{'Model':<20} {'Avg Time (ms)':<15} {'Throughput (samples/s)':<25} {'Memory (MB)':<15}")
print("-" * 60)
print(f"{'Baseline GPT':<20} {baseline_results['avg_time_ms']:<15.2f} {baseline_results['throughput_samples_per_sec']:<25.2f}", end="")
if device.type == "cuda":
    print(f" {baseline_results.get('peak_memory_mb', 'N/A'):<15.2f}")
else:
    print(f" {'N/A':<15}")

print(f"{'Flash Attention':<20} {flash_results['avg_time_ms']:<15.2f} {flash_results['throughput_samples_per_sec']:<25.2f}", end="")
if device.type == "cuda":
    print(f" {flash_results.get('peak_memory_mb', 'N/A'):<15.2f}")
else:
    print(f" {'N/A':<15}")

print("-" * 60)
print(f"\nIMPROVEMENTS:")
print(f"Speed: {speedup:.2f}x faster")
print(f"Throughput: {throughput_improvement:.2f}x higher")
if device.type == "cuda":
    memory_reduction = (1 - flash_results.get('peak_memory_mb', 0) / baseline_results.get('peak_memory_mb', 1)) * 100
    print(f"Memory: {memory_reduction:.1f}% reduction")

# Save detailed results
results = {
    "baseline": baseline_results,
    "flash_attention": flash_results,
    "improvements": {
        "speedup": speedup,
        "throughput_improvement": throughput_improvement,
        "memory_reduction_percent": memory_reduction if device.type == "cuda" else None
    }
}

with open("test/flash_attention_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to test/flash_attention_results.json")

# Verify outputs are identical
print("\nVerifying model outputs...")
baseline_model.eval()
flash_model.eval()

with torch.no_grad():
    x, y = next(iter(valid_loader))
    x = x.to(device)
    
    baseline_logits, _ = baseline_model(x)
    flash_logits, _ = flash_model(x)
    
    # Check if outputs are close (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(baseline_logits - flash_logits)).item()
    print(f"Maximum difference in logits: {max_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ Models produce equivalent outputs!")
    else:
        print("⚠ Warning: Models produce different outputs!")