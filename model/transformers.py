from model.modules.encoder import TransformerEncoder
from model.modules.decoder import TransformerDecoder
from model.modules.positional_encoding import PositionalEncoding
import torch.nn as nn
import torch 

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, vocab_size, context_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, max_len=context_length)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers)
        self.output_linear = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, source, target):
        source_padding_mask = (source == 0)
        target_padding_mask = (target == 0)

        source = self.token_embedding(source)
        source = self.pos_embedding(source)

        target = self.token_embedding(target)
        target = self.pos_embedding(target)

        target_mask = self.generate_causal_mask(target.size(1)).to(target.device)

        memory = self.encoder(source, padding_mask=source_padding_mask)
        output = self.decoder(target, memory, target_mask=target_mask, target_padding_mask=target_padding_mask, memory_padding_mask=source_padding_mask)

        return self.output_linear(output)