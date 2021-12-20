from torchvision.datasets import MNIST
import torchvision
import torch
import albumentations as A
from torchvision.transforms import ConvertImageDtype
import numpy as np


class MNISTClassification(MNIST):
    def __init__(self, atransforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atransforms = atransforms

        self.float = torchvision.transforms.Compose(
            [torchvision.transforms.ConvertImageDtype(torch.float)]
        )

    def __getitem__(self, index: int):
        img: torch.tensor
        img, label = super().__getitem__(index)

        img = img.repeat(3, 1, 1)

        if self.atransforms is not None:
            transformed = self.atransforms(
                image=np.moveaxis(np.array(img.detach().cpu()), 0, -1)
            )
            img = self.float(transformed["image"])

        return img, label
