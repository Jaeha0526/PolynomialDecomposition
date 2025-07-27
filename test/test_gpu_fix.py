#!/usr/bin/python3
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Try to initialize CUDA
try:
    torch.cuda.init()
    print("CUDA initialized successfully")
except Exception as e:
    print(f"CUDA init error: {e}")

# Force device
device = torch.device("cuda:0")
try:
    x = torch.randn(10, 10).to(device)
    print(f"Successfully created tensor on GPU: {x.device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error creating tensor on GPU: {e}")
    
# Try CPU for comparison
device_cpu = torch.device("cpu")
x_cpu = torch.randn(10, 10).to(device_cpu)
print(f"CPU tensor device: {x_cpu.device}")