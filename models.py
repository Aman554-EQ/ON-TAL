import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

# Import our custom Mamba implementation (no external dependencies!)
from mamba_core import MambaEncoder, MambaDecoder


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
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
        

class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_dec_layer=opt["dec_layer"]
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        
        # Mamba-specific hyperparameters (from scratch implementation)
        mamba_state_dim = opt.get("mamba_state_dim", 16)
        mamba_conv_dim = opt.get("mamba_conv_dim", 4)
        mamba_expand = opt.get("mamba_expand", 2)
        
        # FC layers for the 2 streams (RGB and Flow)
        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        # Custom Mamba Encoder - O(L) complexity, no external dependencies!
        self.encoder = MambaEncoder(
            d_model=n_embedding_dim,
            n_layers=n_enc_layer,
            dropout=dropout,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_dim,
            expand=mamba_expand
        )
        
        # Custom Mamba Decoder - with cross-attention to encoder memory
        self.decoder = MambaDecoder(
            d_model=n_embedding_dim,
            n_layers=n_dec_layer,
            dropout=dropout,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_dim,
            expand=mamba_expand
        )
        
        # Classification and Regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(n_embedding_dim, 2)
        )
        
        # Learnable decoder tokens for each anchor
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # inputs - batch x seq_len x featSize
        
        # Process RGB and Flow features separately then concatenate
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        # Permute to seq_len x batch x featsize for encoder
        base_x = base_x.permute([1, 0, 2])
        
        # Add positional encoding
        pe_x = self.positional_encoding(base_x)
        
        # Mamba Encoder
        encoded_x = self.encoder(pe_x)
        
        # Expand decoder tokens for batch
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        
        # Mamba Decoder with cross-attention to encoded features
        decoded_x = self.decoder(decoder_token, encoded_x)
        
        # Permute back to batch x anchors x featsize
        decoded_x = decoded_x.permute([1, 0, 2])
        
        # Classification and regression outputs
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
        
