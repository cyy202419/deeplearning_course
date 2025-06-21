import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from .utils_model import BaseModel

class Transformer(BaseModel):
    def __init__(self, 
                 src_vocab_size, 
                 target_vocab_size, 
                 h_dim,
                 enc_pf_dim,
                 dec_pf_dim,
                 enc_n_layers,
                 dec_n_layers,
                 enc_n_heads,
                 dec_n_heads,
                 enc_dropout,
                 dec_dropout,  **kwargs):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, 
            h_dim, 
            enc_pf_dim, 
            enc_n_heads, 
            enc_n_layers, 
            enc_dropout, 
        )
        self.decoder = Decoder(
            target_vocab_size, 
            h_dim, 
            dec_pf_dim, 
            dec_n_heads, 
            dec_n_layers, 
            dec_dropout, 
        )
        self.fc = nn.Linear(h_dim, target_vocab_size)

    def forward(self, src, target, src_mask, target_mask):

        encoder_output = self.encoder(src, src_mask)

        decoder_output = self.decoder(target, encoder_output, src_mask, target_mask)

        output = self.fc(decoder_output)

        return output