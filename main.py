from PIL import Image

import torchvision.transforms as transforms

from layers import (
    # Identity,
    PatchEmbedding,
    FeedforwardLayer
)

pil2tensor = transforms.PILToTensor()


def main():
    # load sample data
    img = Image.open('./sample.png')
    img = img.resize((224, 224))
    tensor = pil2tensor(img)
    tensor = tensor.unsqueeze(0) / 255.
    # print(tensor)
    # def layers
    patch_embedding = PatchEmbedding(16, 3, 32, dropout=0.)
    mlp = FeedforwardLayer(32)

    # pass to patch_embed
    print('Input', tensor.size())
    tensor = patch_embedding(tensor)
    print('After patch embedding', tensor.size())
    tensor = mlp(tensor)
    print('After MLP', tensor.size())


main()
