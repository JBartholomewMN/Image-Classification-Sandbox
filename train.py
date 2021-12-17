from dataset.coco_dataset import NormalizedCocoDetection
from model.yolov3 import YoloV3
from model.loss import YoloLoss
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from config.config import CFG


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, optimizer, criterion, trainloader, testloader, epochs, device, cfg):
    for epoch in range(epochs):
        for img, boxes, labels in trainloader:
            X = torch.stack(img).to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(X)
                loss = torch.tensor([0]).float().to(device)
                for pred, anch in zip(predictions, cfg["anchors"]):
                    loss += criterion(pred, boxes, labels, anch, cfg)

                # loss = criterion(predictions[0], boxes, labels, cfg["anchors"][0], cfg)

                loss.backward()
                optimizer.step()
                print(loss)


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
