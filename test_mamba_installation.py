#!/usr/bin/env python3
"""
Quick test script to verify Mamba-OAT installation and model forward pass.
Run this after installing dependencies to ensure everything is working.
"""

import torch
import numpy as np
import sys

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from mamba_ssm import Mamba
        print("✓ mamba_ssm imported successfully")
    except ImportError as e:
        print(f"I mamba_ssm not found (using pure-Python fallback): {e}")
        # Not a failure condition for us since we have a fallback
    
    try:
        import einops
        print("✓ einops imported successfully")
    except ImportError as e:
        print(f"I einops not found: {e} (Optional)")
        # return False  <-- Commented out, not required for our implementation
    
    try:
        from models import MYNET, MambaEncoder, MambaDecoder
        print("✓ models.py imported successfully")
        print("✓ MambaEncoder found")
        print("✓ MambaDecoder found")
        print("✓ MYNET found")
    except ImportError as e:
        print(f"✗ Failed to import from models.py: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_model_instantiation():
    """Test model can be created."""
    print("=" * 60)
    print("Testing model instantiation...")
    print("=" * 60)
    
    try:
        import opts_thumos as opts
        from models import MYNET
        
        # Parse default options
        opt = opts.parse_opt()
        opt = vars(opt)
        
        # Parse anchors
        opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]
        
        print(f"Model config:")
        print(f"  - Hidden dim: {opt['hidden_dim']}")
        print(f"  - Encoder layers: {opt['enc_layer']}")
        print(f"  - Decoder layers: {opt['dec_layer']}")
        print(f"  - Mamba state dim: {opt.get('mamba_state_dim', 16)}")
        print(f"  - Mamba conv dim: {opt.get('mamba_conv_dim', 4)}")
        print(f"  - Mamba expand: {opt.get('mamba_expand', 2)}")
        
        # Create model
        model = MYNET(opt)
        print("\n✓ Model instantiated successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return True, model, opt
        
    except Exception as e:
        print(f"\n✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model, opt):
    """Test model forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    
    try:
        batch_size = 2
        seq_len = opt['segment_size']  # 64 by default
        feat_dim = opt['feat_dim']  # 4096 by default
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, feat_dim)
        print(f"Input shape: {dummy_input.shape}")
        print(f"  (batch_size={batch_size}, seq_len={seq_len}, feat_dim={feat_dim})")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            anc_cls, anc_reg = model(dummy_input)
        
        num_anchors = len(opt['anchors'])
        num_classes = opt['num_of_class']
        
        print(f"\nOutput shapes:")
        print(f"  - Classification: {anc_cls.shape}")
        print(f"    Expected: ({batch_size}, {num_anchors}, {num_classes})")
        print(f"  - Regression: {anc_reg.shape}")
        print(f"    Expected: ({batch_size}, {num_anchors}, 2)")
        
        # Verify shapes
        assert anc_cls.shape == (batch_size, num_anchors, num_classes), \
            f"Classification output shape mismatch"
        assert anc_reg.shape == (batch_size, num_anchors, 2), \
            f"Regression output shape mismatch"
        
        print("\n✓ Forward pass successful!")
        print("✓ Output shapes correct!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda_compatibility():
    """Test CUDA availability and compatibility."""
    print("\n" + "=" * 60)
    print("Testing CUDA compatibility...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available - will use CPU (slower)")
        print("  Training on CPU will be very slow. GPU recommended.")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MAMBA-OAT INSTALLATION TEST")
    print("=" * 60)
    print()
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test CUDA
    test_cuda_compatibility()
    
    # Test model instantiation
    success, model, opt = test_model_instantiation()
    if not success:
        print("\n✗ Model instantiation failed.")
        sys.exit(1)
    
    # Test forward pass
    if not test_forward_pass(model, opt):
        print("\n✗ Forward pass failed.")
        sys.exit(1)
    
    # All tests passed
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nYou're ready to train the Mamba-OAT model!")
    print("\nNext steps:")
    print("  1. Download THUMOS'14 features to data/")
    print("  2. Run: python main.py --mode=train")
    print("  3. Train OSN: python supnet.py --mode=make/train")
    print("  4. Test: python main.py --mode=test")
    print()


if __name__ == "__main__":
    main()
