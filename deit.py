import torch.nn as nn

from layers import (
    PatchEmbedding,
    EncoderLayer
)


class DeiT(nn.Module):

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
            dropout=dropout,
            use_cls_token=True, use_distill_token=True
        )
        self.encoders = [
            EncoderLayer(
                embed_dim=embed_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_dropout=attn_dropout, mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout
            ) for _ in range(num_encoders)
        ]

        self.head = nn.Linear(embed_dim, num_classes)
        self.head_distill = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [N, C, h, w]
        x = self.patch_embed(x)  # x: [N, embed_dim, h'*w']

        for encoder in self.encoders:
            x = encoder(x)

        # index 0 for cls token
        x_cls = self.head(x[:, 0])
        # index 1 for distill token
        x_distill = self.head_distill(x[:, 1]).contiguous()

        # TODO: return format not identical, bad
        if self.training:
            return {
                'logit': x_cls,
                'distill': x_distill
            }
        else:
            return {
                'logit': (x_cls + x_distill) / 2
            }
