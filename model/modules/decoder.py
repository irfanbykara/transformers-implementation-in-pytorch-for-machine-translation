from model.modules.attention import MultiHeadSelfAttention
from model.modules.feed_forward import FeedForward
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, target_mask=None, target_padding_mask=None, memory_padding_mask=None):
        x = x + self.self_attn(x, x, x, mask=target_mask, key_padding_mask=target_padding_mask)
        x = self.norm1(x)
        x = x + self.cross_attn(x, memory, memory, key_padding_mask=memory_padding_mask)
        x = self.norm2(x)
        x = x + self.ff(x)
        x = self.norm3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, target_mask=None, target_padding_mask=None, memory_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, target_mask, target_padding_mask, memory_padding_mask)
        return self.norm(x)