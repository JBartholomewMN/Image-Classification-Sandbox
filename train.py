from torchvision.datasets import CocoDetection, VOCDetection, ImageNet
from model.yolov3 import YoloV3
from model.loss import YoloLoss
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import yaml

def collate_fn(batch):
    return torch.tensor(torch.stack([b[0] for b in batch])), [b[1] for b in batch]

def train(model, optimizer, criterion, trainloader, testloader, epochs):
    for epoch in range(epochs):
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = criterion(predictions, labels, None)

if __name__ == "__main__":
    
    # load config data from yaml
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = YoloV3(config="config/config.yaml")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    tfs = transforms.Compose([
        transforms.Resize([416, 416]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    coco_train_data = CocoDetection(
        "dataset/train2017/", 
        "dataset/annotations_trainval2017/annotations/instances_train2017.json",
        transform=tfs,
        
    )
    trainloader = torch.utils.data.DataLoader(coco_train_data, batch_size=5, shuffle=True, num_workers=4, collate_fn=collate_fn)
    testloader = CocoDetection("dataset/val2017/", "dataset/annotations_trainval2017/annotations/instances_val2017.json")
    criterion = YoloLoss()
    train(model, optimizer, criterion, trainloader, testloader, 1)