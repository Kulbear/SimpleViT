from PIL import Image

import torchvision.transforms as transforms

from layers import (
    # Identity,
    PatchEmbedding,
    FeedforwardLayer,
    MultiHeadSelfAttention
)

from vit import ViT

pil2tensor = transforms.PILToTensor()


def main():
    # load sample data
    img = Image.open('./sample.png')
    img = img.resize((224, 224))
    tensor = pil2tensor(img)
    tensor = tensor.unsqueeze(0) / 255.
    ipt = tensor
    print(tensor.size())

    # def layers
    patch_embedding = PatchEmbedding(16, 3, 32, dropout=0.)
    mlp = FeedforwardLayer(32)
    mh_attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4, qkv_bias=False, qk_scale=None)
    vit = ViT()

    # pass to patch_embed
    print('Input', tensor.size())
    tensor = patch_embedding(tensor)
    print('After patch embedding', tensor.size())
    tensor = mlp(tensor)
    print('After MLP', tensor.size())
    tensor = mh_attn(tensor)
    print('After attention', tensor.size())


    print('Input', ipt.size())
    x = vit(ipt)
    print('After ViT', x.size())


main()
