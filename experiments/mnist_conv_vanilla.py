import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
from torchvision.transforms.transforms import ToTensor
from dataset.coco_dataset import CocoDetection
from dataset.imagenet import ImageNetClassifcation
from dataset.MNIST import MNISTClassification
from model.model import Backbone
import os


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch]),
    )


"""
    Building blocks available to you:
    --------------------------------------
    2d convolution, residual block, convolutional transformer, transformer

    To define each, follow these examples.

    2D convolution
    ------------------------------------------------------------ 
        3 in channels, 64 out channels, 5 kernel size, 1 stride:
        Example: ["conv", 3, 64, 5, 1]


    Residual Conv Block
    ------------------------------------------------------------
        Repeated 8 times:
            First layer:
                256 in channels, 512 out channels, 3 kernel size
            Second layer:
                512 in channels, 256 out channels, 3 kernel size

        Example: ["res", [256, 512], [512, 256], [3, 3], 8, False]


    Convolutional Transformer
    ------------------------------------------------------------
        256 in channels, 96 filters for each (k, q, v), kqv kernel size 3, 8 attention heads
        Example: ["convformer", 256, 96, 3, 8, False]


    Convolutional Transformer
    ------------------------------------------------------------
        256 in channels, 128 vector size (k, q, v), 8 attention heads
        Example: ["transformer", 256, 128, 8, False]
    

"""

CFG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "layers": {
        1: ["conv", 3, 16, 3, 1],
        2: ["conv", 16, 32, 3, 1],
        3: ["conv", 32, 64, 3, 2],
        4: ["conv", 64, 128, 3, 2],
    },
    "size": [28, 28],
    "T_transforms": transforms.Compose(
        [
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ]
    ),
    "A_transforms": A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    ),
    "A_transforms_test": A.Compose(
        [
            ToTensorV2(),
        ]
    ),
    "optimizer": torch.optim.Adam,
    "optimizer_args": {"lr": 0.001},
    "dataset": MNISTClassification,
    "trainset_args": {"root": "dataset/data/MNISTClassification/", "download": True},
    "trainloader": torch.utils.data.DataLoader,
    "testloader": torch.utils.data.DataLoader,
    "loss_accumulations": 4,
    "trainloader_args": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "testset_args": {
        "root": "dataset/data/MNISTClassification/",
        "download": True,
        "train": False,
    },
    "testloader_args": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "epochs": 100,
    "nclasses": 10,
    "topkaccuracy": 2,
    "model": Backbone,
    "model_inits": [Backbone.add_classifier],
    "weights_save_path": os.path.basename(__file__) + ".pt",
    "criterion": torch.nn.CrossEntropyLoss(),
}
