import torch
import torch.nn as nn
from mynet_arch import MYNET
from opts_thumos import parse_opt
import argparse

def test_mynet_consistency():
    print("Testing MYNET Forward vs Step Consistency...")
    
    # Mock Opts
    opt = {
        "feat_dim": 32, # small for test
        "num_of_class": 5,
        "hidden_dim": 16,
        "enc_layer": 2,
        "dec_layer": 2,
        "anchors": [4, 8],
        "mamba_state_dim": 8,
        "mamba_conv_dim": 4, 
        "mamba_expand": 2
    }
    
    model = MYNET(opt)
    model.eval() # Disable dropout
    
    # Create Dummy Input: (B, L, F)
    B, L, F = 1, 10, 32
    inputs = torch.randn(B, L, F)
    
    # 1. Forward Pass (Whole Sequence)
    with torch.no_grad():
        pred_cls_fwd, pred_reg_fwd = model(inputs)
        
    # 2. Step Pass (Frame by Frame)
    with torch.no_grad():
        caches = model.allocate_inference_cache(B, inputs.device)
        
        # Loop through sequence
        for t in range(L):
            x_t = inputs[:, t, :] # (B, F)
            print(f"Time {t}: Input shape {x_t.shape}")
            
            # Step
            last_cls, last_reg, caches = model.step(x_t, caches, t)
            print(f"Time {t}: Step Output shape {last_cls.shape}")
            
    diff_cls = (pred_cls_fwd - last_cls).abs().max()
    diff_reg = (pred_reg_fwd - last_reg).abs().max()
    
    print(f"Max Diff CLS: {diff_cls}")
    print(f"Max Diff REG: {diff_reg}")
    
    if diff_cls < 1e-4 and diff_reg < 1e-4:
        print("✅ MYNET Consistency Test Passed")
        return True
    else:
        print("❌ MYNET Consistency Test Failed")
        return False

if __name__ == "__main__":
    test_mynet_consistency()
