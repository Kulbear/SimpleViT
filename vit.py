import torch.nn as nn

from layers import (
    FeedforwardLayer,
    MultiHeadSelfAttention,
    PatchEmbedding
)


class Encoder(nn.Module):

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


class ViT(nn.Module):

    def __init__(
            self,
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            num_encoders=5,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            dropout=0.,
            qk_scale=None,
            attn_dropout=0.,
            mlp_dropout=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=224, patch_size=16,
            in_channels=in_channels, embed_dim=embed_dim,
            dropout=dropout
        )
        self.encoders = [
            Encoder(
                embed_dim=embed_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_dropout=attn_dropout, mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout
            ) for _ in range(num_encoders)
        ]

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [N, C, h, w]
        x = self.patch_embed(x)  # x: [N, embed_dim, h'*w']
        for encoder in self.encoders:
            x = encoder(x)
        x = self.classifier(x[:, -1, :])

        return x
