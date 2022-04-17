import torch
import torch.nn as nn


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PatchEmbedding(nn.Module):

    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            dropout=0.
    ):
        super().__init__()

        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1, 1, embed_dim,
                requires_grad=True
            )
        )

        self.position_embedding = nn.Parameter(
            torch.normal(
                0.,
                0.02,
                size=(1, n_patches + 1, embed_dim),  # +1 for cls token
                requires_grad=True
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [N, C, h, w]
        x = self.embedding(x)  # [N, embed_dim, h', w']

        x = torch.flatten(x, start_dim=2)
        x = x.transpose(2, 1)  # [N, h'*w', embed_dim]

        # need to add cls_token for entire batch
        # cls_token: [1, 1, embed_dim]
        # cls_tokens: [N, 1, embed_dim] where N is batch size
        cls_tokens = self.cls_token.expand([x.shape[0], 1, -1])
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.position_embedding  # just broadcast
        x = self.dropout(x)

        return x  # [N, h'*w', embed_dim]


class FeedforwardLayer(nn.Module):

    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()  # by the paper
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

    def __init__(
            self,
            embed_dim=768,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            dropout=0.
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = int(self.embed_dim / self.num_heads)
        self.all_heads_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim, self.all_heads_dim * 3, bias=False if qkv_bias is False else None)

        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_heads_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _transpose_multi_head(self, x):
        # B: batch size, N: num_patches
        # x: [B, N, all_heads_dim]
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)  # [B, N, num_heads, head_dim]
        x = x.reshape(new_shape)
        # x: [B, N, num_heads, head_dim]
        # tensor.transpose is better :)
        # but still can do x = x.permute((0, 2, 1, 3))
        x = x.transpose(2, 1)
        # x: [B, num_heads, N, head_dim]
        return x

    def forward(self, x):
        # B: batch size, N: # of patches + cls token if needed
        print(1, x.shape)
        # x: [B, N, all_heads_dim]
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        print(2, qkv[0].shape)
        # [B, N, all_heads_dim] * 3
        q, k, v = map(self._transpose_multi_head, qkv)

        # q, k, v: [B, num_heads, N, head_dim]
        attn = torch.matmul(q, k.transpose(2, 3))  # q * k'
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # attn: [B, num_heads, N, N]
        # every patch to every other patch attention, must be an N-by-N matrix

        # out: [B, num_heads, N, N] * [N, head_dim]
        out = torch.matmul(attn, v)
        # out: [B, num_heads, N, head_dim]
        out = out.permute((0, 2, 1, 3))
        out = out.reshape([B, N, -1])

        # out: [B, N, all_heads_dim]
        out = self.proj(out)
        out = self.dropout(out)

        return out
