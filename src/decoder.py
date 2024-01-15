import torch.nn as nn
import src.components as c

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout) -> None:
        super().__init__()

        self.self_attn = c.MultiHeadAttention(d_model, num_heads)
        self.cross_attn = c.MultiHeadAttention(d_model, num_heads)
        self.feed_forward = c.FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x