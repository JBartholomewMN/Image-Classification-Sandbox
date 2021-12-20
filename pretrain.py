from dataset.imagenet import ImageNetClassifcation
from model.yolov3 import Backbone

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn.functional
import torch.nn as nn
from config.config import CFG
import tqdm
from model.eval import _evaluate


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch]),
    )


def train(model, optimizer, criterion, trainloader, testloader, epochs, device, cfg):

    model.train()
    running_loss = 0.0
    for epoch in range(epochs):
        for i, (img_batch, label_batch) in enumerate(
            tqdm.tqdm(trainloader, desc=f"Training Epoch {epoch}")
        ):

            optimizer.zero_grad()
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            predictions = model(img_batch)

            loss = criterion(predictions, label_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                print(label_batch)
                print(predictions.argmax(1))


if __name__ == "__main__":

    model = Backbone(CFG)
    model.add_classifier()
    model = model.to(CFG["device"])
    print("model has %d params" % sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=CFG["lr"], momentum=CFG["momentum"]
    )

    im_train_data = ImageNetClassifcation(
        atransforms=CFG["A_transforms"],
        root="dataset/data/imagenet_train",
        transform=CFG["T_transforms"],
    )
    trainloader = torch.utils.data.DataLoader(
        im_train_data,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["nworkers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
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
