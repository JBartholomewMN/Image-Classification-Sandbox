from torchvision.datasets import CocoDetection
import torchvision
import torch
import albumentations as A
import numpy as np


class NormalizedCocoDetection(CocoDetection):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms if transforms is not None else None

    def __getitem__(self, index: int):
        img: torch.tensor
        img, labels = super().__getitem__(index)

        device = img.device

        # pull out the bboxes and labels we need
        # discard the rest
        bbs = np.array([det["bbox"] for det in labels])
        labels = np.array([[det["image_id"]] + [det["category_id"]] for det in labels])

        img = np.moveaxis(img.detach().numpy(), 0, -1)
        transformed = self.transform(image=img, bboxes=bbs, class_labels=labels)

        transformed["bboxes"] = A.augmentations.bbox_utils.normalize_bboxes(
            transformed["bboxes"], img.shape[0], img.shape[1]
        )

        # labels vector is (id, class, x, y, w, h)
        # print(labels.shape[0], len(transformed["bboxes"]))
        labels = np.hstack((labels, transformed["bboxes"]))

        return torch.movedim(torch.tensor(transformed["image"]), -1, 0), torch.tensor(
            labels
        )
