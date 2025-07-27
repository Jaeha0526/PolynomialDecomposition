#!/usr/bin/python3
"""
Test script to benchmark the current model with ON example dataset
This will serve as our baseline before implementing Flash Attention
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add path to mingpt
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
import dataset

# Set device - force GPU if available
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Try to force CUDA device
try:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    # Test if we can actually use it
    test_tensor = torch.randn(1, 1).to(device)
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"GPU initialization failed: {e}")
    device = torch.device("cpu")
    print(f"Falling back to: {device}")

# Configuration matching example_with_ON_data.sh
BLOCK_SIZE = 400
N_EMBD = 256
N_LAYER = 4
N_HEAD = 8
BATCH_SIZE = 32  # Reduced for CPU testing
MAX_NUMBER_TOKEN = 101

# Define vocabulary (matching run.py)
chars_symbolic = [
    "□",
    "a","b","c","d","e","x","y","z",
    "⁇","?",
    "a0","a1","b0","b1",
    "N","P","&","+","*","^",
] + [str(i) for i in range(0, 10)]

# Extend vocabulary for ON dataset tokens
for i in range(2, 21):  # a2-a20, b2-b20
    chars_symbolic.extend([f"a{i}", f"b{i}"])

vocab_size = len(chars_symbolic)
print(f"Vocabulary size: {vocab_size}")

# Load datasets
print("\nLoading ON datasets...")
train_data_path = "data_storage/dataset/example_ON/ON_data_train.txt"
valid_data_path = "data_storage/dataset/example_ON/ON_data_valid.txt"

with open(train_data_path, 'r') as f:
    train_data = f.read()
with open(valid_data_path, 'r') as f:
    valid_data = f.read()

train_dataset = dataset.SymbolicDataset(BLOCK_SIZE, chars_symbolic, train_data)
valid_dataset = dataset.SymbolicDataset(BLOCK_SIZE, chars_symbolic, valid_data)

print(f"Train dataset size: {len(train_dataset):,}")
print(f"Valid dataset size: {len(valid_dataset):,}")

# Create model
model_config = GPTConfig(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD
)

print("\nCreating model...")
model = GPT(model_config)
model = model.to(device)

# Create data loaders for benchmarking
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Benchmark forward pass
print("\n" + "="*50)
print("BASELINE PERFORMANCE TEST")
print("="*50)

def benchmark_forward_pass(model, loader, num_batches=10):
    """Benchmark forward pass performance"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Warm up GPU
            if i == 0:
                for _ in range(3):
                    _ = model(x)
                torch.cuda.synchronize() if device.type == "cuda" else None
            
            # Time the forward pass
            start = time.time()
            logits, loss = model(x, targets=y)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.time()
            
            times.append(end - start)
            
            if i == 0:
                print(f"Batch shape: {x.shape}")
                print(f"Loss: {loss.item():.4f}")
    
    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])  # Exclude first batch
    else:
        avg_time = times[0] if times else 0
    throughput = BATCH_SIZE / avg_time if avg_time > 0 else 0
    
    return avg_time, throughput, times

# Run benchmark
print("\nRunning forward pass benchmark...")
avg_time, throughput, times = benchmark_forward_pass(model, valid_loader, num_batches=20)

print(f"\nResults (excluding warmup):")
print(f"Average forward pass time: {avg_time*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Min time: {min(times[1:])*1000:.2f} ms")
print(f"Max time: {max(times[1:])*1000:.2f} ms")

# Memory usage
if device.type == "cuda":
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

# Test attention computation specifically
print("\n" + "="*50)
print("ATTENTION LAYER BENCHMARK")
print("="*50)

# Get a sample batch
x, y = next(iter(valid_loader))
x = x.to(device)

# Benchmark just the attention layers
attention_times = []
model.eval()

with torch.no_grad():
    # Get embeddings
    token_embeddings = model.tok_emb(x)
    position_embeddings = model.pos_emb[:, :x.size(1), :]
    x_emb = model.drop(token_embeddings + position_embeddings)
    
    # Warmup
    for _ in range(3):
        h = x_emb
        for block in model.blocks:
            h = block(h)
    torch.cuda.synchronize() if device.type == "cuda" else None
    
    # Benchmark attention through all blocks
    for i in range(10):
        start = time.time()
        h = x_emb
        for block in model.blocks:
            h = block(h)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.time()
        attention_times.append(end - start)

avg_attention_time = sum(attention_times) / len(attention_times)
print(f"Average time through all {N_LAYER} attention blocks: {avg_attention_time*1000:.2f} ms")
print(f"Average time per attention block: {avg_attention_time*1000/N_LAYER:.2f} ms")

# Save baseline results
results = {
    "model_params": sum(p.numel() for p in model.parameters()),
    "avg_forward_time_ms": avg_time * 1000,
    "throughput_samples_per_sec": throughput,
    "avg_attention_time_ms": avg_attention_time * 1000,
    "device": str(device),
    "batch_size": BATCH_SIZE,
    "sequence_length": BLOCK_SIZE,
    "n_layers": N_LAYER,
    "n_heads": N_HEAD,
    "n_embd": N_EMBD
}

print("\n" + "="*50)
print("BASELINE RESULTS SUMMARY")
print("="*50)
for key, value in results.items():
    print(f"{key}: {value}")

# Save results for comparison
import json
with open("test/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nBaseline results saved to test/baseline_results.json")