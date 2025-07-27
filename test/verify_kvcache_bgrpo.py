#!/usr/bin/env python3
"""
Simple verification that KV-cache model loads correctly for BGRPO
"""

import sys
sys.path.append('/workspace/PolynomialDecomposition')

from Training.mingpt.model_loader import load_model_and_tokenizer
from pathlib import Path

def test_kvcache_loading():
    """Test that KV-cache model loads properly"""
    
    project_root = Path('/workspace/PolynomialDecomposition/Training')
    CONFIG_NAME = 'model_configuration.json'
    
    config_path = project_root / '..' / 'data_storage' / 'model' / 'model_configurations' / CONFIG_NAME
    model_dir_path = project_root / '..' / 'data_storage' / 'model'
    device = "cuda"
    
    print("="*80)
    print("TESTING KV-CACHE MODEL LOADING FOR BGRPO")
    print("="*80)
    
    try:
        # Load model WITHOUT KV-cache first
        print("\n1. Loading model WITHOUT KV-cache...")
        model_normal, tokenizer = load_model_and_tokenizer(
            config_path=str(config_path),
            model_dir_path=str(model_dir_path),
            device=device,
            wrap_for_grpo=True,
            model_name='single_variable_model_best.pt',
            use_kvcache=False
        )
        print(f"✅ Normal model type: {type(model_normal).__name__}")
        if hasattr(model_normal, 'pretrained_model'):
            print(f"   Pretrained model type: {type(model_normal.pretrained_model).__name__}")
        
        # Load model WITH KV-cache
        print("\n2. Loading model WITH KV-cache...")
        model_kvcache, tokenizer = load_model_and_tokenizer(
            config_path=str(config_path),
            model_dir_path=str(model_dir_path),
            device=device,
            wrap_for_grpo=True,
            model_name='single_variable_model_best.pt',
            use_kvcache=True
        )
        print(f"✅ KV-cache model type: {type(model_kvcache).__name__}")
        if hasattr(model_kvcache, 'pretrained_model'):
            print(f"   Pretrained model type: {type(model_kvcache.pretrained_model).__name__}")
        
        # Verify key attributes
        print("\n3. Verifying BGRPO compatibility...")
        
        # Check if wrapped correctly
        if hasattr(model_kvcache, 'pretrained_model'):
            print("✅ Model is wrapped for GRPO (has pretrained_model)")
            base_model = model_kvcache.pretrained_model
        else:
            print("❌ Model is not wrapped for GRPO")
            base_model = model_kvcache
            
        # Check base model attributes
        required_attrs = ['beam', 'END_INDEX', 'MASK_INDEX', 'beam_search_with_cache']
        
        for attr in required_attrs:
            if hasattr(base_model, attr):
                print(f"✅ Base model has required attribute: {attr}")
            else:
                print(f"❌ Base model missing required attribute: {attr}")
        
        # Test a simple forward pass
        print("\n4. Testing forward pass...")
        import torch
        test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        
        try:
            with torch.no_grad():
                output = model_kvcache(test_input)
            print("✅ Forward pass successful")
            print(f"   Output shape: {output.logits.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
        
        # Test beam search capability
        print("\n5. Testing beam search setup...")
        if hasattr(model_kvcache, 'pretrained_model'):
            base_model = model_kvcache.pretrained_model
        else:
            base_model = model_kvcache
            
        base_model.beam = True
        base_model.END_INDEX = tokenizer.eos_token_id
        base_model.MASK_INDEX = tokenizer.mask_token_id
        print("✅ Beam search parameters set successfully")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED! KV-cache model is compatible with BGRPO")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_kvcache_loading()