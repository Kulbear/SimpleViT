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
            dropout=0.,
            use_norm=False,
            use_cls_token=True,
            use_distill_token=False
    ):
        super().__init__()

        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.use_cls_token = use_cls_token
        self.use_distill_token = use_distill_token
        self.use_norm = use_norm
        self.embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.norm = nn.LayerNorm(embed_dim)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(
                    1, 1, embed_dim,
                    requires_grad=True
                )
            )
            n_patches += 1  # +1 for cls token
        if self.use_distill_token:
            self.distill_token = nn.Parameter(
                torch.zeros(
                    1, 1, embed_dim,
                    requires_grad=True
                )
            )
            n_patches += 1  # +1 for distill token
            # initialize to non-zeros
            nn.init.xavier_uniform(self.distill_token.data)

        self.position_embedding = nn.Parameter(
            torch.normal(
                0.,
                0.02,
                size=(1, n_patches, embed_dim),
                requires_grad=True
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [N, C, h, w]
        x = self.embedding(x)  # [N, embed_dim, h', w']

        x = torch.flatten(x, start_dim=2)
        x = x.transpose(2, 1)  # [N, h'*w', embed_dim]

        if self.use_norm:
            x = self.norm(x)

        # need to add extra token for entire batch
        if self.use_distill_token:
            # distill_token: [1, 1, embed_dim]
            # distill_tokens: [N, 1, embed_dim] where N is batch size
            distill_tokens = self.distill_token.expand([x.shape[0], 1, -1])
            x = torch.cat([distill_tokens, x], dim=1)

        if self.use_cls_token:
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


class EncoderLayer(nn.Module):

    def __init__(
            self,
            embed_dim=768,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_dropout=0.,
            mlp_ratio=4.0,
            mlp_dropout=0.
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            dropout=attn_dropout
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = FeedforwardLayer(
            embed_dim,
            mlp_ratio=mlp_ratio, dropout=mlp_dropout
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

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
        # B: batch size, N: # of patches + (cls token + distill token) if needed
        # x: [B, N, all_heads_dim]
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
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
