import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.ff(x)
