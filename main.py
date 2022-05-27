from PIL import Image
from deit import DeiT

import torchvision.transforms as transforms

from layers import (
    # Identity,  # no longer used
    PatchEmbedding,
    PatchMerging,
    FeedforwardLayer,
    MultiHeadSelfAttention,
    SwinBlock
)

from vit import ViT
from deit import DeiT

pil2tensor = transforms.PILToTensor()


def main():
    """ main function for testing everything now """
    # load sample data
    img = Image.open('./sample.png')
    img = img.resize((224, 224))
    tensor = pil2tensor(img)
    tensor = tensor.unsqueeze(0) / 255.
    dummy = tensor
    print(tensor.size())

    # def layers
    patch_embedding = PatchEmbedding(
        image_size=224, patch_size=16,
        in_channels=3, embed_dim=768,
        dropout=0.
    )
    mlp = FeedforwardLayer(
        768,
        mlp_ratio=4.0, dropout=0.
    )
    mh_attn = MultiHeadSelfAttention(
        embed_dim=768, num_heads=8,
        qkv_bias=False, qk_scale=None
    )

    swin_patch_embedding = PatchEmbedding(
        image_size=224, patch_size=4,
        in_channels=3, embed_dim=96,
        dropout=0.,
        use_norm=True,
        use_cls_token=False,
        use_distill_token=False
    )
    swin_block = SwinBlock(embed_dim=96, input_resolution=(56, 56),
                           num_heads=4, window_size=7)
    patch_merging = PatchMerging(input_resolution=[56, 56], embed_dim=96)

    vit = ViT()
    deit = DeiT()

    # pass to patch_embed
    print('Input', tensor.size())
    tensor = patch_embedding(tensor)
    print('After patch embedding', tensor.size())
    tensor = mlp(tensor)
    print('After MLP', tensor.size())
    tensor = mh_attn(tensor)
    print('After attention', tensor.size())

    ipt = dummy.expand((4, -1, -1, -1))
    print('Input', ipt.size())
    x = vit(ipt)['logit']
    print('After ViT', x.size())
    x = deit(ipt)['logit']
    print('After DeiT', x.size())

    swin_ipt = dummy.expand((2, -1, -1, -1))
    print('Swin Input', swin_ipt.size())
    tensor = swin_patch_embedding(swin_ipt)
    print('After Swin patch embedding', tensor.size())
    tensor = swin_block(tensor)
    print('After Swin block', tensor.size())
    tensor = patch_merging(tensor)
    print('After patch merging', tensor.size())


main()
