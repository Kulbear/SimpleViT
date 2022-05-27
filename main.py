from PIL import Image
from deit import DeiT

import torchvision.transforms as transforms

from layers import (
    # Identity,  # no longer used
    PatchEmbedding,
    FeedforwardLayer,
    MultiHeadSelfAttention
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
    ipt = tensor
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

    ipt = ipt.expand((4, -1, -1, -1))
    print('Input', ipt.size())
    x = vit(ipt)['logit']
    print('After ViT', x.size())
    x = deit(ipt)['logit']
    print('After DeiT', x.size())


main()
