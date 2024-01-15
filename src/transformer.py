import torch
import torch.nn as nn
import src.encoder as enc 
import src.decoder as dec
import src.components as c

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout) -> None:
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = c.PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([enc.EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([dec.DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, src, tgt, device):
        # [batch_size, 1, 1, src_seq_length]
        # src mask is used to prevent the model from attending to padding tokens in the source sequence. 
        # Padding tokens are typically represented by 0.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # tgt mask masks out padding tokens in the target sequence. 
        # prevents positions from attending to subsequent positions, which is crucial during training to maintain the autoregressive property (a token should not see future tokens)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        # lower triangle of 1
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt, device):
        src_mask, tgt_mask = self.generate_mask(src, tgt, device) # [batch, 1, 1, seq_length]
        # Create embeddings
        src_embedded = self.dropout(self.encoder_embed(src)) # [batch, seq_length, d_model]
        tgt_embedded = self.dropout(self.decoder_embed(tgt))

        src_embedded = self.pe(src_embedded) # [batch, seq_length, d_model]
        tgt_embedded = self.pe(tgt_embedded)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output