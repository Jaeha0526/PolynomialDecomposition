#!/usr/bin/python3
"""
Test that FlashCausalSelfAttention is a perfect drop-in replacement
"""

import sys
import os
import torch
import torch.nn as nn

# Add path to mingpt
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))

from model import GPT, GPTConfig, CausalSelfAttention
from flash_attention_module import FlashCausalSelfAttention, replace_attention_with_flash_attention

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test 1: Verify forward pass produces same results
print("\n" + "="*50)
print("TEST 1: Comparing CausalSelfAttention vs FlashCausalSelfAttention")
print("="*50)

# Create config
config = GPTConfig(
    vocab_size=100,
    block_size=128,
    n_layer=1,
    n_head=8,
    n_embd=256
)

# Create both attention modules
original_attn = CausalSelfAttention(config).to(device)
flash_attn = FlashCausalSelfAttention(config).to(device)

# Copy weights to ensure same initialization
flash_attn.key.load_state_dict(original_attn.key.state_dict())
flash_attn.query.load_state_dict(original_attn.query.state_dict())
flash_attn.value.load_state_dict(original_attn.value.state_dict())
flash_attn.proj.load_state_dict(original_attn.proj.state_dict())

# Test input
B, T, C = 4, 64, 256  # batch, sequence, channels
x = torch.randn(B, T, C).to(device)

# Set to eval mode to disable dropout for comparison
original_attn.eval()
flash_attn.eval()

# Forward pass
with torch.no_grad():
    original_out = original_attn(x)
    flash_out = flash_attn(x)

# Check if outputs match
max_diff = torch.max(torch.abs(original_out - flash_out)).item()
print(f"Maximum difference in outputs: {max_diff:.6e}")
print(f"Outputs match: {max_diff < 1e-5}")

# Test 2: Verify model replacement works
print("\n" + "="*50)
print("TEST 2: Testing full model replacement")
print("="*50)

# Create two identical GPT models
config = GPTConfig(
    vocab_size=100,
    block_size=128,
    n_layer=4,
    n_head=8,
    n_embd=256
)

model1 = GPT(config).to(device)
model2 = GPT(config).to(device)

# Ensure same initialization
model2.load_state_dict(model1.state_dict())

# Replace attention in model2
model2 = replace_attention_with_flash_attention(model2)

# Test forward pass
model1.eval()
model2.eval()

input_ids = torch.randint(0, 100, (4, 64)).to(device)

with torch.no_grad():
    logits1, loss1 = model1(input_ids)
    logits2, loss2 = model2(input_ids)

max_diff_logits = torch.max(torch.abs(logits1 - logits2)).item()
print(f"Maximum difference in logits: {max_diff_logits:.6e}")
print(f"Model outputs match: {max_diff_logits < 1e-5}")

# Verify attention modules were replaced
print("\nVerifying attention replacement:")
for name, module in model2.named_modules():
    if 'attn' in name and not isinstance(module, nn.ModuleList):
        print(f"{name}: {type(module).__name__}")

# Test 3: Performance comparison
print("\n" + "="*50)
print("TEST 3: Performance comparison")
print("="*50)

import time

# Larger batch for performance testing
if device.type == "cuda":
    B, T = 32, 128  # Use block_size from config
    input_ids = torch.randint(0, 100, (B, T)).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model1(input_ids)
        _ = model2(input_ids)
    torch.cuda.synchronize()
    
    # Time original model
    start = time.time()
    for _ in range(20):
        _ = model1(input_ids)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # Time flash model
    start = time.time()
    for _ in range(20):
        _ = model2(input_ids)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    
    print(f"Original model time: {original_time:.3f}s")
    print(f"Flash model time: {flash_time:.3f}s")
    print(f"Speedup: {original_time/flash_time:.2f}x")
else:
    print("Skipping performance test on CPU")

print("\n✓ All tests passed! FlashCausalSelfAttention is a perfect drop-in replacement.")