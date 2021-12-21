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
    "layers": {
        # if conv: ["conv", inchans, outchans, ksize, stride]
        # if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
        # if convtrans: ["convtrans", inchans, kqvchans, nheads, hiddensize, outsize, repeats, store_output]
        # if transformer: ["transformer", in/out chans, kqvchans, nheads, store_output]
        1: ["conv", 3, 128, 1, 1],
        2: ["transformer", 128, 128, 8, False],
        3: ["transformer", 128, 128, 8, False],
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
    "trainloader_args": {
        "batch_size": 24,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn,
        "pin_memory": True,
    },
    "testset_args": {"root": "dataset/data/MNISTClassification/", "train": False},
    "testloader_args": {
        "batch_size": 24,
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
    "weights_save_path": "mnist_xformer_vanilla.pt",
    "criterion": torch.nn.CrossEntropyLoss(),
}
