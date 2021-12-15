from torchvision.datasets import CocoDetection, VOCDetection, ImageNet
from model.yolov3 import YoloV3
from model.loss import YoloLoss
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import yaml

def collate_fn(batch):

    return tuple(zip(*batch))

def train(model, optimizer, criterion, trainloader, testloader, epochs, device):
    for epoch in range(epochs):
        for X, y in trainloader:
            X = torch.stack(X).to(device)
            optimizer.zero_grad()

            predictions = model(X)
            # compute the loss over every output head...
            loss = torch.sum([criterion(p, y, None) for p in predictions])

if __name__ == "__main__":
    
    # load config data from yaml
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YoloV3(config="config/config.yaml").to(device)
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
    train(model, optimizer, criterion, trainloader, testloader, 1, device)