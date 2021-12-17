from dataset.coco_dataset import NormalizedCocoDetection
from model.yolov3 import YoloV3
from model.loss import YoloLoss
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from config.config import CFG


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.cat([b[1] for b in batch], dim=0)


def train(
    model, optimizer, criterion: YoloLoss, trainloader, testloader, epochs, device, cfg
):
    for epoch in range(epochs):
        for img_batch, label_batch in trainloader:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(img_batch.to(device))
                loss, stats = criterion(predictions, label_batch, cfg["anchors"], cfg)

                # loss = criterion(predictions[0], boxes, labels, cfg["anchors"][0], cfg)

                loss.backward()
                optimizer.step()
                print(stats)


if __name__ == "__main__":

    model = YoloV3(CFG).to(CFG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    coco_train_data = NormalizedCocoDetection(
        CFG["A_transforms"],
        "dataset/train2017/",
        "dataset/annotations_trainval2017/annotations/instances_train2017.json",
        transform=CFG["T_transforms"],
    )
    trainloader = torch.utils.data.DataLoader(
        coco_train_data,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["nworkers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    criterion = YoloLoss()
    train(
        model,
        optimizer,
        criterion,
        trainloader,
        None,
        CFG["epochs"],
        CFG["device"],
        CFG,
    )
