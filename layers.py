import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size,
                 in_channels, embed_dim,
                 dropout=0.):
        super().__init__()

        self.embedding = nn.Conv2d(in_channels,
                                   embed_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   bias=False)
        self.flatten = nn.Flatten(start_dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [b, c, h, w]
        x = self.embedding(x)  # [b, embed_dim, h', w']

        x = self.flatten(x)  # [b, embed_dim, h'*w']
        x = x.transpose(2, 1)  # [b, h'*w', embed_dim]

        x = self.dropout(x)

        return x


class FeedforwardLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
