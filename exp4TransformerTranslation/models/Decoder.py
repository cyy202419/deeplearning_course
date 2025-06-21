import torch
import torch.nn as nn
from .utils_model import *

class Decoder(BaseModel):
    def __init__(self, vocab_size, h_dim, pf_dim, n_heads, n_layers, dropout, max_seq_len=200):
        super().__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.word_embeddings = WordEmbedding(vocab_size, h_dim)

        self.pe = PositionEmbedding(max_seq_len, h_dim)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([h_dim])).to(self.device)
        
        for _ in range(n_layers):
            self.layers.append(DecoderLayer(h_dim, pf_dim, n_heads, dropout))
    
    def forward(self, target, encoder_output, src_mask, target_mask):

        output = self.word_embeddings(target) * self.scale
        tar_len = target.shape[1]
        pos = torch.arange(0, tar_len).unsqueeze(0).repeat(target.shape[0], 1).to(self.device)
        output = self.dropout(output + self.pe(pos))
        for i in range(self.n_layers):
            output = self.layers[i](output, encoder_output, src_mask, target_mask)
        
        return output
    

class DecoderLayer(BaseModel):
    def __init__(self, h_dim, pf_dim, n_heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(h_dim, n_heads, dropout)
        self.attention = MultiHeadAttention(h_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(h_dim, pf_dim, dropout)
        
        self.self_attention_layer_norm = nn.LayerNorm(h_dim)
        self.attention_layer_norm = nn.LayerNorm(h_dim)
        self.ff_layer_norm = nn.LayerNorm(h_dim)
        
        self.self_attention_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, target, encoder_output, src_mask, target_mask):
        # Self attention (masked)
        self_attention_output = self.self_attention(target, target, target, target_mask)
        output = self.self_attention_layer_norm(target + self.self_attention_dropout(self_attention_output))
        
        # Encoder-decoder attention
        attention_output = self.attention(output, encoder_output, encoder_output, src_mask)
        output = self.attention_layer_norm(output + self.attention_dropout(attention_output))
        
        # Position-wise feedforward
        ff_output = self.positionwise_feedforward(output)
        output = self.ff_layer_norm(output + self.ff_dropout(ff_output))
        
        return output