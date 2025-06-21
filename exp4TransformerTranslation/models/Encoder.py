import torch
import torch.nn as nn
from .utils_model import *



class Encoder(BaseModel):
    def __init__(self, vocab_size, h_dim, pf_dim, n_heads, n_layers, dropout, max_seq_len=200):
        super().__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        
        # Embedding layers
        self.word_embedding = WordEmbedding(vocab_size, h_dim)
        self.pe = PositionEmbedding(max_seq_len, h_dim)
        
        # Encoder layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EncoderLayer(h_dim, n_heads, pf_dim, dropout))
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([h_dim])).to(self.device)
    
    def forward(self, src, src_mask):
        # Word embedding with scaling
        output = self.word_embedding(src) * self.scale
        
        # Positional encoding
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)
        output = self.dropout(output + self.pe(pos))
        
        # Transformer encoder layers
        for i in range(self.n_layers):
            output = self.layers[i](output, src_mask)
        
        return output

class EncoderLayer(BaseModel):
    def __init__(self, h_dim, n_heads, pf_dim, dropout):
        super().__init__()
        # Multi-head attention layer
        self.attention = MultiHeadAttention(h_dim, n_heads, dropout)
        
        # Layer normalization components
        self.attention_layer_norm = nn.LayerNorm(h_dim)
        self.ff_layer_norm = nn.LayerNorm(h_dim)
        
        # Position-wise feedforward network
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            h_dim, pf_dim, dropout
        )
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # 1. Multi-head attention with residual connection
        att_output = self.attention(src, src, src, src_mask)
        output = self.attention_layer_norm(
            src + self.attention_dropout(att_output)
        )
        
        # 2. Position-wise feedforward with residual connection
        ff_output = self.positionwise_feedforward(output)
        output = self.ff_layer_norm(
            output + self.ff_dropout(ff_output)
        )
        
        return output
    
    
    