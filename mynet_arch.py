"""
MYNET Architecture with Quick Fixes for mAP 50+
Key improvements over original:
1. Cross-attention in decoder (attend to encoder memory)
2. Gated branch fusion (instead of simple concat)
3. Learnable anchor embeddings
"""
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

try:
    from mamba_core import MambaEncoder, MambaBlock, CausalCrossAttention, RMSNorm
except ImportError:
    from mamba_core import MambaEncoder, MambaBlock, CausalCrossAttention, RMSNorm


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
    def step(self, token_embedding, time_step):
        pos = self.pos_embedding[time_step, :].unsqueeze(0)
        return token_embedding + pos


class GatedFusion(nn.Module):
    """FIX #2: Gated fusion instead of simple concatenation"""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x_short, x_long):
        x_cat = torch.cat([x_short, x_long], dim=-1)
        gate = self.gate(x_cat)
        proj = self.proj(x_cat)
        # Gated combination: gate * short + (1-gate) * long + residual
        out = gate * x_short + (1 - gate) * x_long + proj
        return self.norm(out)


class MultiScaleMambaEncoder(nn.Module):
    """Dual-path encoder with gated fusion"""
    def __init__(self, d_model, n_layers, d_state=16, expand=2, dropout=0.1):
        super().__init__()
        self.short_term_branch = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=3, expand=expand)
            for _ in range(n_layers)
        ])
        self.long_term_branch = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state*2, d_conv=9, expand=expand)
            for _ in range(n_layers)
        ])
        # FIX #2: Replace simple concat+linear with gated fusion
        self.fusion = GatedFusion(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_short = x
        for layer in self.short_term_branch:
            x_short = layer(x_short)
        x_long = x
        for layer in self.long_term_branch:
            x_long = layer(x_long)
        # Gated fusion instead of simple concat
        x_out = self.fusion(x_short, x_long)
        x_out = self.dropout(x_out)
        return x_out

    def step(self, x, caches):
        x_short = x
        short_caches = caches['short']
        for i, layer in enumerate(self.short_term_branch):
            x_short, short_caches[i] = layer.step(x_short, short_caches[i])
        x_long = x
        long_caches = caches['long']
        for i, layer in enumerate(self.long_term_branch):
            x_long, long_caches[i] = layer.step(x_long, long_caches[i])
        x_out = self.fusion(x_short, x_long)
        return x_out, {'short': short_caches, 'long': long_caches}

    def allocate_inference_cache(self, batch_size, device):
        return {
            'short': [layer.allocate_inference_cache(batch_size, device) for layer in self.short_term_branch],
            'long': [layer.allocate_inference_cache(batch_size, device) for layer in self.long_term_branch]
        }


class CrossAttentionProposalDecoder(nn.Module):
    """
    FIX #1: Decoder with cross-attention to encoder memory
    This is the BIGGEST fix - allows decoder to attend to rich encoder features
    """
    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2, dropout=0.1, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        # Cross-attention to encoder memory (KEY FIX!)
        self.cross_attn = nn.ModuleList([
            CausalCrossAttention(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.cross_attn_norm = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory):
        """
        x: decoder input (L, B, D) or (B, L, D)
        memory: encoder output to attend to (B, L, D)
        """
        # Ensure (B, L, D) format
        if x.dim() == 3 and x.size(0) != memory.size(0):
            x = x.transpose(0, 1)  # (L, B, D) -> (B, L, D)
            
        for mamba_layer, cross_layer, cross_norm in zip(self.layers, self.cross_attn, self.cross_attn_norm):
            # Self-attention via Mamba
            x = mamba_layer(x)
            
            # Cross-attention to encoder memory (KEY!)
            x_residual = x
            x_normed = cross_norm(x)
            cross_out = cross_layer(x_normed, memory, use_causal_mask=True)
            x = x_residual + cross_out
            x = self.dropout(x)
            
        x = self.norm(x)
        return x
    
    def step(self, x, memory, caches):
        """Online inference step"""
        for i, (mamba_layer, cross_layer, cross_norm) in enumerate(
            zip(self.layers, self.cross_attn, self.cross_attn_norm)):
            x, caches[i] = mamba_layer.step(x, caches[i])
            
            # Cross-attention (attend to all past encoder memory)
            x_residual = x
            x_normed = cross_norm(x.unsqueeze(1))
            cross_out = cross_layer(x_normed, memory, use_causal_mask=False)
            x = x_residual + cross_out.squeeze(1)
            
        x = self.norm(x)
        return x, caches
    
    def allocate_inference_cache(self, batch_size, device):
        return [layer.allocate_inference_cache(batch_size, device) for layer in self.layers]


class MYNET(torch.nn.Module):
    """
    Improved MYNET with three key fixes:
    1. Cross-attention in decoder to attend to encoder memory
    2. Gated fusion for branch combination
    3. Learnable anchor embeddings
    """
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"] 
        n_dec_layer = opt["dec_layer"] 
        self.anchors = opt["anchors"]
        dropout = 0.3
        self.best_map = 0
        
        mamba_state_dim = opt.get("mamba_state_dim", 16)
        mamba_conv_dim = opt.get("mamba_conv_dim", 4)
        mamba_expand = opt.get("mamba_expand", 2)
        
        # Feature reduction for RGB and Flow
        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.norm_rgb = nn.LayerNorm(n_embedding_dim//2)
        self.norm_flow = nn.LayerNorm(n_embedding_dim//2)
        
        # Dual-path encoders with gated fusion
        self.encoder_rgb = MultiScaleMambaEncoder(n_embedding_dim//2, n_enc_layer, mamba_state_dim, mamba_expand, dropout)
        self.encoder_flow = MultiScaleMambaEncoder(n_embedding_dim//2, n_enc_layer, mamba_state_dim, mamba_expand, dropout)
        
        # FIX #2: Gated cross-modal fusion
        self.stream_fusion = GatedFusion(n_embedding_dim//2)
        self.stream_proj = nn.Linear(n_embedding_dim//2, n_embedding_dim)
        
        # FIX #1: Decoder with cross-attention to encoder memory
        self.proposal_decoder = CrossAttentionProposalDecoder(
            n_embedding_dim, n_dec_layer, mamba_state_dim, mamba_conv_dim, mamba_expand, dropout
        )
        
        num_anchors = len(self.anchors)
        self.num_anchors = num_anchors
        
        # FIX #3: Learnable anchor embeddings
        self.anchor_embeddings = nn.Parameter(torch.randn(num_anchors, n_embedding_dim) * 0.02)
        
        # Classification and regression heads (per-anchor, outputs n_class and 2 respectively)
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(n_embedding_dim, n_class)  # Per-anchor: outputs (B, num_anchors, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(n_embedding_dim, 2)  # Per-anchor: outputs (B, num_anchors, 2)
        )
        
        self.pe = PositionalEncoding(n_embedding_dim, dropout, maxlen=3000)

    def forward(self, inputs):
        B, L, _ = inputs.shape
        
        # Split RGB and Flow
        rgb = inputs[:,:,:self.n_feature//2]
        flow = inputs[:,:,self.n_feature//2:]
        
        # Project and normalize
        x_rgb = self.norm_rgb(self.feature_reduction_rgb(rgb))
        x_flow = self.norm_flow(self.feature_reduction_flow(flow))
        
        # Encode with dual-path Mamba
        enc_rgb = self.encoder_rgb(x_rgb)  # (B, L, D/2)
        enc_flow = self.encoder_flow(x_flow)  # (B, L, D/2)
        
        # FIX #2: Gated cross-modal fusion
        x_fused = self.stream_fusion(enc_rgb, enc_flow)  # (B, L, D/2)
        x_fused = self.stream_proj(x_fused)  # (B, L, D)
        
        # Add positional encoding (transpose for PE which expects L, B, D)
        x_fused = x_fused.transpose(0, 1)  # (L, B, D)
        x_fused = self.pe(x_fused)
        x_fused = x_fused.transpose(0, 1)  # Back to (B, L, D)
        
        # Store encoder memory for cross-attention
        encoder_memory = x_fused  # (B, L, D)
        
        # FIX #1: Decoder with cross-attention to encoder memory
        decoded = self.proposal_decoder(x_fused, encoder_memory)  # (B, L, D)
        
        # FIX #3: Add anchor embeddings to decoded features (at last position)
        last_decoded = decoded[:, -1, :]  # (B, D)
        # Expand anchor embeddings for batch
        anchor_embeds = self.anchor_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, num_anchors, D)
        
        # Combine decoded features with anchor embeddings
        last_decoded_expanded = last_decoded.unsqueeze(1).expand(-1, self.num_anchors, -1)  # (B, num_anchors, D)
        anchor_features = last_decoded_expanded + anchor_embeds  # (B, num_anchors, D)
        
        # Classification and regression per anchor (apply to each anchor independently)
        anc_cls = self.classifier(anchor_features)  # (B, num_anchors, num_class)
        anc_reg = self.regressor(anchor_features)   # (B, num_anchors, 2)
        
        return anc_cls, anc_reg

    def step(self, x, caches, time_step):
        """Online inference: process single frame"""
        rgb = x[:, :self.n_feature//2]
        flow = x[:, self.n_feature//2:]
        
        x_rgb = self.norm_rgb(self.feature_reduction_rgb(rgb))
        x_flow = self.norm_flow(self.feature_reduction_flow(flow))
        
        x_rgb, caches['encoder_rgb'] = self.encoder_rgb.step(x_rgb, caches['encoder_rgb'])
        x_flow, caches['encoder_flow'] = self.encoder_flow.step(x_flow, caches['encoder_flow'])
        
        x_fused = self.stream_fusion(x_rgb, x_flow)
        x_fused = self.stream_proj(x_fused)
        x_fused = self.pe.step(x_fused, time_step)
        
        # Update encoder memory cache
        if 'encoder_memory' not in caches:
            caches['encoder_memory'] = x_fused.unsqueeze(1)
        else:
            caches['encoder_memory'] = torch.cat([caches['encoder_memory'], x_fused.unsqueeze(1)], dim=1)
        
        # Decoder with cross-attention to accumulated encoder memory
        x_decoded, caches['decoder'] = self.proposal_decoder.step(
            x_fused, caches['encoder_memory'], caches['decoder']
        )
        
        # Add anchor embeddings
        B = x.size(0)
        anchor_embeds = self.anchor_embeddings.unsqueeze(0).expand(B, -1, -1)
        x_decoded_expanded = x_decoded.unsqueeze(1).expand(-1, self.num_anchors, -1)
        anchor_features = x_decoded_expanded + anchor_embeds
        
        # Apply classifier/regressor per-anchor
        cls_out = self.classifier(anchor_features)  # (B, num_anchors, n_class)
        reg_out = self.regressor(anchor_features)   # (B, num_anchors, 2)
        
        return cls_out, reg_out, caches

    def allocate_inference_cache(self, batch_size, device):
        return {
            'encoder_rgb': self.encoder_rgb.allocate_inference_cache(batch_size, device),
            'encoder_flow': self.encoder_flow.allocate_inference_cache(batch_size, device),
            'decoder': self.proposal_decoder.allocate_inference_cache(batch_size, device)
        }
