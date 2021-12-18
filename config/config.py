import albumentations as A
import torchvision.transforms as transforms
import torch

DATASET = "coco"
CFG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "layers": {
        # if conv: ["conv", inchans, outchans, ksize, stride]
        # if res: ["res", [inchans...], [outchans...], [ksizes...], repeats, store_output]
        1: ["conv", 3, 32, 3, 1],
        2: ["conv", 32, 64, 3, 2],
        3: ["res", [64, 32], [32, 64], [1, 3], 1, False],
        4: ["conv", 64, 128, 3, 2],
        5: ["res", [128, 64], [64, 128], [1, 3], 2, False],
        6: ["conv", 128, 256, 3, 2],
        7: ["res", [256, 128], [128, 256], [1, 3], 8, True],
        8: ["conv", 256, 512, 3, 2],
        9: ["res", [512, 256], [256, 512], [1, 3], 8, True],
        10: ["conv", 512, 1024, 3, 2],
        11: ["res", [1024, 512], [512, 1024], [1, 3], 4, True],
    },
    "nclasses": 91,
    "nanchors": 3,
    "nbbvals": 5,
    "lr": 0.0001,
    "batch_size": 20,
    "nworkers": 1,
    "epochs": 100,
    "iou_thresh": 0.5,
    "conf_thresh": 0.6,
    "nms_thresh": 0.3,
    "checkpoint_file": "checkpoint.pth.tar",
    "anchors": [
        [[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]],
        [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
        [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],
    ],
    "size": [416, 416],
    "T_transforms": transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    ),
    "A_transforms": A.Compose(
        [A.Resize(416, 416)],
        bbox_params=A.BboxParams(format=DATASET, label_fields=["class_labels"]),
    ),
}
