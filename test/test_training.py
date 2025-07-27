#!/usr/bin/env python3
"""
Quick test script to verify the current model training works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Training/mingpt'))

import torch
import numpy as np
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
import dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic data for testing
def create_test_data(num_samples=100, max_length=50):
    """Create simple synthetic polynomial-like data for testing"""
    # Using a simple pattern: "a + b ? c" where c = a + b (single digits only)
    data_lines = []
    for _ in range(num_samples):
        a = np.random.randint(1, 5)  # Keep numbers small
        b = np.random.randint(1, 5)  # So sum is single digit
        result = a + b
        # Create a simple expression
        line = f"{a} + {b} ? {result}"
        data_lines.append(line)
    return "\n".join(data_lines)

# Generate test data
print("Generating test data...")
train_data = create_test_data(200)
valid_data = create_test_data(50)

# Define vocabulary (matching the one in run.py)
chars_symbolic = [
    "□",
    "a","b","c","d","e","x","y","z",
    "⁇","?",
    "a0","a1","b0","b1",
    "N","P","&","+","*","^",
] + [str(i) for i in range(0, 10)]

# Create datasets
block_size = 128
print(f"Creating datasets with block_size={block_size}")
train_dataset = dataset.SymbolicDataset(
    block_size,
    chars_symbolic,
    train_data
)
valid_dataset = dataset.SymbolicDataset(
    block_size,
    chars_symbolic,
    valid_data
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")

# Create model
vocab_size = len(chars_symbolic)
print(f"Vocabulary size: {vocab_size}")

model_config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=2,      # Small model for testing
    n_head=4,       # 4 heads
    n_embd=128      # Small embedding dimension
)

print("Creating model...")
model = GPT(model_config)
model = model.to(device)

# Training configuration
train_config = TrainerConfig(
    max_epochs=2,  # Just 2 epochs for testing
    batch_size=16,  # Small batch size for testing
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=16 * 100 * block_size,  # Adjusted for small dataset
    num_workers=0,  # Set to 0 for testing
    ckpt_path=None,  # No checkpoint for test
    shuffle=True,
    weight_decay=0.1
)

print("\nTraining configuration:")
print(f"- Epochs: {train_config.max_epochs}")
print(f"- Batch size: {train_config.batch_size}")
print(f"- Learning rate: {train_config.learning_rate}")

# Create trainer and start training
print("\nStarting training...")
trainer = Trainer(model, train_dataset, valid_dataset, train_config)

try:
    trainer.train()
    print("\n✓ Training completed successfully!")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    
    # Create a test input
    test_input = "2 + 3 ⁇"
    print(f"Test input: {test_input}")
    
    # Convert to tokens
    tokens = [train_dataset.stoi.get(s, train_dataset.stoi["□"]) for s in test_input.split()]
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        logits, _ = model(x)
        # Get the last token prediction
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_tokens = torch.topk(probs, k=5)
        
        print("Top 5 predicted tokens:")
        for i, (prob, idx) in enumerate(zip(top_tokens.values, top_tokens.indices)):
            token = train_dataset.itos[idx.item()]
            print(f"  {i+1}. '{token}' (prob: {prob.item():.3f})")
    
    print("\n✓ Model inference works correctly!")
    
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")