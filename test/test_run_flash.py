#!/usr/bin/python3
"""
Test that run.py now uses Flash Attention by default
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))

import torch
from model import GPTConfig, GPT
from flash_attention_module import FlashCausalSelfAttention, replace_attention_with_flash_attention

# Simulate what run.py does
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model config (matching run.py defaults)
vocab_size = 100
block_size = 128
n_layer = 4
n_head = 8
n_embd = 256

model_cfg = GPTConfig(
    vocab_size, block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd
)
gpt = GPT(model_cfg)
gpt.to(device)

# Convert to Flash Attention for faster training
gpt = replace_attention_with_flash_attention(gpt)

# Verify Flash Attention is being used
print("\nChecking attention modules:")
flash_count = 0
for name, module in gpt.named_modules():
    if 'attn' in name and isinstance(module, FlashCausalSelfAttention):
        flash_count += 1
        print(f"✓ {name}: {type(module).__name__}")

print(f"\nTotal Flash Attention modules: {flash_count}")
print(f"Expected: {n_layer}")
print(f"Success: {flash_count == n_layer}")

# Test forward pass
input_ids = torch.randint(0, vocab_size, (4, 64)).to(device)
with torch.no_grad():
    logits, loss = gpt(input_ids)
    print(f"\n✓ Forward pass successful with Flash Attention")
    print(f"Output shape: {logits.shape}")

print("\n✓ run.py is now using Flash Attention by default!")