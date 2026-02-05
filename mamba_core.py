"""
Mamba: Selective State Space Models - Refactored Implementation
Focus: Online Temporal Action Localization (Causal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import optimized Mamba implementation, fall back to pure-Python if it fails.
try:
    from mamba_ssm import Mamba as OptimizedMamba
    USE_OPTIMIZED_MAMBA = True
    print("Successfully imported optimized Mamba from mamba_ssm")
except ImportError:
    USE_OPTIMIZED_MAMBA = False
    print("Using pure-Python SelectiveSSM fallback.")

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class SelectiveSSM(nn.Module):
    """Selective State Space Model for Mamba (User Provided / Standard S6)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(d_model // 16, 1) if dt_rank=="auto" else dt_rank
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Causal Conv1d: padding=d_conv-1 guarantees we can slice to causality
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialization
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias for better initial stability (preserving history)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state+1, dtype=torch.float32).repeat(self.d_inner,1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # Conv1d needs (B, C, L)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L] # Causal slice
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        x_proj = self.x_proj(x)
        dt, B_param, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        y = self.selective_scan(x, dt, A, B_param, C)
        
        y = y + x * self.D
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y
    
    def selective_scan(self, x, dt, A, B, C):
        """
        Memory-efficient selective scan using chunked processing.
        Reduces gradient graph size to avoid OOM.
        """
        B_batch, L, d_inner = x.shape
        d_state = self.d_state
        
        # Chunk size for memory efficiency (smaller = less memory but slower)
        chunk_size = min(32, L)  # Process 32 timesteps at a time
        
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            chunk_ys = []
            
            for i in range(chunk_start, chunk_end):
                dt_i = dt[:, i, :]
                dA_i = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))
                B_i = B[:, i, :]
                dB_i = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)
                x_i = x[:, i, :].unsqueeze(-1)
                
                h = dA_i * h + dB_i * x_i
                
                C_i = C[:, i, :].unsqueeze(1)
                y_i = (h * C_i).sum(dim=-1)
                chunk_ys.append(y_i)
            
            # Stack chunk results
            ys.extend(chunk_ys)
            
            # Detach h every chunk to limit gradient graph depth
            # This trades off exact gradients for memory efficiency
            if chunk_end < L:
                h = h.detach()
            
        return torch.stack(ys, dim=1)
    
    def step(self, x, cache):
        """
        Single step forward for online inference.
        x: (B, D) - single frame
        cache: dict with 'conv_state' and 'ssm_state'
        Returns: y (B, D), updated cache
        """
        conv_state = cache['conv_state']  # (B, d_inner, d_conv)
        ssm_state = cache['ssm_state']    # (B, d_inner, d_state)
        
        # Project input
        xz = self.in_proj(x)  # (B, d_inner * 2)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, d_inner)
        
        # Update conv state and apply convolution
        conv_state = torch.roll(conv_state, -1, dims=2)
        conv_state[:, :, -1] = x_inner
        x_conv = (conv_state * self.conv1d.weight.squeeze(2)).sum(dim=2) + self.conv1d.bias
        x_conv = F.silu(x_conv)
        
        # Compute SSM parameters
        x_proj = self.x_proj(x_conv)  # (B, dt_rank + 2*d_state)
        dt, B_param, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # SSM step
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(1)  # (B, d_inner, d_state)
        
        ssm_state = dA * ssm_state + dB * x_conv.unsqueeze(-1)
        y = (ssm_state * C.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        
        # Output
        y = y + x_conv * self.D
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        cache['conv_state'] = conv_state
        cache['ssm_state'] = ssm_state
        
        return y, cache
    
    def allocate_inference_cache(self, batch_size, device):
        """Allocate cache for online inference"""
        return {
            'conv_state': torch.zeros(batch_size, self.d_inner, self.d_conv, device=device),
            'ssm_state': torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        }


class MambaBlock(nn.Module):
    """Single Mamba block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        
        if USE_OPTIMIZED_MAMBA:
            self.mamba = OptimizedMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.mamba = SelectiveSSM(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            )

    def forward(self, x):
        return x + self.mamba(self.norm(x))
    
    def step(self, x, cache):
        """Single step for online inference"""
        y, cache = self.mamba.step(self.norm(x), cache)
        return x + y, cache
    
    def allocate_inference_cache(self, batch_size, device):
        """Allocate cache for Mamba block"""
        if USE_OPTIMIZED_MAMBA:
            # OptimizedMamba has its own cache format
            return {
                'conv_state': torch.zeros(batch_size, self.mamba.d_inner, self.mamba.d_conv, device=device),
                'ssm_state': torch.zeros(batch_size, self.mamba.d_inner, self.mamba.d_state, device=device)
            }
        else:
            return self.mamba.allocate_inference_cache(batch_size, device)


class MambaEncoder(nn.Module):
    """
    Mamba-based encoder: stack of Mamba blocks.
    Strictly Causal for Online TAL capabilities.
    """
    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (L, B, D) - Seq First from models.py
        x = x.transpose(0, 1) # (B, L, D)
        
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
            
        x = self.norm(x)
        x = x.transpose(0, 1) # Back to (L, B, D)
        return x


class CausalCrossAttention(nn.Module):
    """
    Causal cross-attention for online temporal action localization.
    Decoder (Query) attends to Encoder (Key/Value).
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, query, key_value, use_causal_mask=True):
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if use_causal_mask:
            # Full Causal Mask: Q[i] attends to K[0...i]
            mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        return self.out_proj(out)


class MambaDecoder(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Cross Attention to Encoder Memory
        self.cross_attn = nn.ModuleList([
            CausalCrossAttention(d_model=d_model, num_heads=8, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.cross_attn_norm = nn.ModuleList([
            RMSNorm(d_model)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # tgt: (T, B, D), memory: (S, B, D)
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        for mamba_layer, cross_layer, cross_norm in zip(self.layers, self.cross_attn, self.cross_attn_norm):
            tgt = mamba_layer(tgt)
            
            tgt_residual = tgt
            tgt_normed = cross_norm(tgt)
            # Decoder Tokens (Anchors) attend to ALL Encoder Memory
            # For Online: Ensure Memory is strictly Past/Current (handled by Encoder & Data)
            cross_out = cross_layer(tgt_normed, memory, use_causal_mask=False)
            
            tgt = tgt_residual + cross_out
            tgt = self.dropout(tgt)
            
        tgt = self.norm(tgt)
        return tgt.transpose(0, 1)

if __name__ == "__main__":
    print("Testing Updated Mamba Implementation...")
    B, L, D = 2, 64, 32
    enc = MambaEncoder(D, 2)
    x = torch.randn(L, B, D)
    y = enc(x)
    print(f"Encoder Output: {y.shape}")
    assert y.shape == x.shape
    print("✅ Encoder Test Passed")
