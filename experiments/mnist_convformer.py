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


# if conv: ["conv", inchans, outchans, ksize, stride]
# if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
# if convformer: ["convtrans", inchans, kqvchans, kernsize, nheads, store_output]
# if transformer: ["transformer", in/out chans, kqvchans, nheads, store_output]

CFG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "layers": {
        1: ["conv", 3, 128, 1, 1],
        2: ["convformer", 128, 128, 3, 8, False],
        3: ["convformer", 128, 128, 3, 8, False],
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
    # "optimizer_args": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
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
    "testset_args": {"root": "dataset/data/MNISTClassification/", "train": False},
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
    "weights_save_path": "mnist_convformer.pt",
    "criterion": torch.nn.CrossEntropyLoss(),
}
