import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
from torchvision.transforms.transforms import ToTensor
from dataset.coco_dataset import CocoDetection
from dataset.imagenet import ImageNetClassifcation
from dataset.MNIST import MNISTClassification
from model.yolov3 import Backbone


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch]),
    )


CFG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    # "layers": {
    #     # if conv: ["conv", inchans, outchans, ksize, stride]
    #     # if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
    #     1: ["conv", 3, 32, 3, 1],
    #     2: ["conv", 32, 64, 3, 2],
    #     3: ["res", [64, 32], [32, 64], [1, 3], 1, False],
    #     4: ["conv", 64, 128, 3, 2],
    #     5: ["res", [128, 64], [64, 128], [1, 3], 2, False],
    #     6: ["conv", 128, 256, 3, 2],
    #     7: ["res", [256, 128], [128, 256], [1, 3], 8, True],
    #     8: ["conv", 256, 512, 3, 2],
    #     9: ["res", [512, 256], [256, 512], [1, 3], 8, True],
    #     10: ["conv", 512, 1024, 3, 2],
    #     11: ["res", [1024, 512], [512, 1024], [1, 3], 4, True],
    # },
    # "layers": {
    #     # if conv: ["conv", inchans, outchans, ksize, stride]
    #     # if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
    #     1: ["conv", 3, 32, 3, 1],
    #     2: ["conv", 32, 64, 3, 2],
    #     4: ["conv", 64, 128, 3, 2],
    #     6: ["conv", 128, 256, 3, 2],
    #     8: ["conv", 256, 512, 3, 2],
    #     10: ["conv", 512, 1024, 3, 2],
    # },
    "layers": {
        # if conv: ["conv", inchans, outchans, ksize, stride]
        # if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
        # if convtrans: ["convtrans", inchans, kqvchans, nheads, hiddensize, outsize, repeats, store_output]
        1: ["conv", 3, 32, 3, 1],
        2: ["conv", 32, 64, 3, 2],
        3: ["res", [64, 32], [32, 64], [1, 3], 1, False],
        4: ["conv", 64, 128, 3, 2],
        5: ["res", [128, 64], [64, 128], [1, 3], 2, False],
        6: ["conv", 128, 256, 3, 2],
        7: ["res", [256, 128], [128, 256], [1, 3], 8, True],
        8: ["conv", 256, 512, 3, 2],
        9: ["convformer", 512, 96, 8, 512, 512, 1, True],
        10: ["conv", 512, 1024, 3, 2],
        11: ["convformer", 1024, 96, 8, 1024, 1024, 1, True],
    },
    "nanchors": 3,
    "nbbvals": 5,
    "iou_thresh": 0.5,
    "conf_thresh": 0.5,
    "nms_thresh": 0.3,
    "anchors": [
        [[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]],
        [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
        [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],
    ],
    "size": [416, 416],
    "T_transforms": transforms.Compose(
        [
            transforms.PILToTensor(),
            # transforms.ConvertImageDtype(torch.float),
        ]
    ),
    "A_transforms": A.Compose(
        [
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A.Resize(416, 416),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    ),
    "A_transforms_test": A.Compose(
        [
            A.Normalize(),
            A.Resize(416, 416),
            ToTensorV2(),
        ]
    ),
    "optimizer": torch.optim.SGD,
    "optimizer_args": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
    "dataset": MNISTClassification,
    "trainset_args": {"root": "dataset/data/MNISTClassification/", "download": True},
    "trainloader": torch.utils.data.DataLoader,
    "testloader": torch.utils.data.DataLoader,
    "trainloader_args": {
        "batch_size": 12,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "testset_args": {"root": "dataset/data/MNISTClassification/", "train": False},
    "testloader_args": {
        "batch_size": 12,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "epochs": 100,
    "nclasses": 10,
    "model": Backbone,
    "model_inits": [Backbone.add_classifier],
    "weights_save_path": "exp2.pt",
    "criterion": torch.nn.CrossEntropyLoss(),
}
