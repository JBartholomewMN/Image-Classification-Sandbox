import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
from torchvision.transforms.transforms import ToTensor
from dataset.coco_dataset import CocoDetection
from dataset.imagenet import ImageNetClassifcation
from dataset.MNIST import MNISTClassification
from model.yolov3 import Backbone
import os


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch]),
    )


# if conv: ["conv", inchans, outchans, ksize, stride]
# if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
# if convformer: ["convtrans", inchans, kqvchans, kernsize, nheads, store_output]
# if transformer: ["transformer", in/out chans, kqvchans, nheads, store_output]

CFG = {
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "layers": {
        1: ["conv", 3, 32, 3, 1],
        2: ["conv", 32, 64, 3, 2],
        3: ["conv", 64, 128, 4, 2],
        4: ["conv", 128, 256, 4, 2],
        5: ["convformer", 256, 128, 1, 8, False],
        6: ["conv", 256, 512, 4, 2],
        7: ["convformer", 512, 128, 1, 8, False],
        8: ["conv", 512, 1024, 3, 2],
        9: ["convformer", 1024, 128, 1, 8, False],
    },
    "size": [412, 412],
    "T_transforms": transforms.Compose(
        [
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ]
    ),
    "A_transforms": A.Compose(
        [
            A.Normalize(),
            A.Resize(412, 412),
            A.HorizontalFlip(p=0.5),
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
            A.Normalize(),
            A.Resize(412, 412),
            ToTensorV2(),
        ]
    ),
    "optimizer": torch.optim.Adam,
    # "optimizer_args": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
    "optimizer_args": {"lr": 0.001},
    "dataset": ImageNetClassifcation,
    "trainset_args": {"root": "dataset/data/", "split": "train"},
    "trainloader": torch.utils.data.DataLoader,
    "testloader": torch.utils.data.DataLoader,
    "loss_accumulations": 16,
    "trainloader_args": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 8,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "testset_args": {"root": "dataset/data/", "split": "val"},
    "testloader_args": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 8,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "epochs": 250,
    "nclasses": 1000,
    "topkaccuracy": 5,
    "model": Backbone,
    "model_inits": [Backbone.add_classifier],
    "weights_save_path": os.path.basename(__file__)+".pt",
    "criterion": torch.nn.CrossEntropyLoss(),
}
