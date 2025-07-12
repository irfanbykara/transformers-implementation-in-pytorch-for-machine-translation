from model.modules.attention import MultiHeadSelfAttention
from model.modules.feed_forward import FeedForward
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.final_norm(x)
