import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, query, key, value, mask=None, key_padding_mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, value)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        B, L_q = query.shape[:2]
        L_kv = key.shape[1]

        Q = self.Q(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.K(key).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.V(value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        output = self.scaled_dot_product_attention(Q, K, V, mask, key_padding_mask)
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        return self.output_proj(output)