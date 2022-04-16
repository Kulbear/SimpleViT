import torch
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(
            embed_dim,
            self.all_head_dim * 3,
            bias=False if qkv_bias is False else None)

        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _transpose_multi_head(self, x):
        # x: [B, N, all_head_dim]
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        x = x.permute((0, 2, 1, 3))
        # x: [B, num_heads, N, head_dim]
        return x

    def forward(self, x):
        # B: batch size, N: # of patches
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self._transpose_multi_head, qkv)

        # q, k, v: [B, num_heads, N, head_dim]
        attn = torch.matmul(q, k.transpose(2, 3))  # q * k'
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # attn: [B, num_heads, N, N]
        # every patch to every other patch attention, must be an N-by-N matrix

        out = torch.matmul(attn, v)
        out = out.permute((0, 2, 1, 3))
        # attn: [B, N, num_heads, head_dim]
        out = out.reshape([B, N, -1])

        out = self.proj(out)
        out = self.dropout(out)

        return out
