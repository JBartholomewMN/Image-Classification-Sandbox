from typing import Optional
from model.yolov3 import Backbone
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch.nn.functional
import torch.nn as nn
import tqdm
from model.eval import _evaluate
import torchmetrics
import importlib.util
import argparse
import os
from config.build_config import build_config


def train(model, optimizer, criterion, trainloader, testloader, epochs, device, cfg):

    top1accbest = 0
    topkaccbest = 0
    t1metric = torchmetrics.Accuracy()
    tkmetric = torchmetrics.Accuracy(top_k=cfg["topkaccuracy"])
    running_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, (img_batch, label_batch) in enumerate(
            tqdm.tqdm(trainloader, desc=f"Training Epoch {epoch}")
        ):

            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            predictions = model(img_batch)

            loss = criterion(predictions, label_batch) / cfg["loss_accumulations"]
            loss.backward()

            # weights update
            if ((i + 1) % cfg["loss_accumulations"] == 0) or (
                i + 1 == len(trainloader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * cfg["loss_accumulations"]
            if i % 100 == 99:
                print(
                    "[%d, %5d] training loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / 100)
                )
                running_loss = 0.0

        model.eval()
        for timgb, tlabs in tqdm.tqdm(testloader, desc="Testing"):
            with torch.no_grad():
                preds = model(timgb.to(device)).detach().cpu()
                acc1 = t1metric(preds, tlabs)
                acck = tkmetric(preds, tlabs)

        acc1 = t1metric.compute()
        acck = tkmetric.compute()

        if acc1 > top1accbest:
            top1accbest = acc1
            topkaccbest = acck
            torch.save(model.state_dict(), cfg["weights_save_path"])

        print("Epoch Top-1 accuracy: ", acc1)
        print("Epoch Top-%d accuracy: " % cfg["topkaccuracy"], acck)
        print("Best  Top-1 accuracy: ", top1accbest)
        print("Best  Top-%d accuracy: " % cfg["topkaccuracy"], topkaccbest)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to configuration (.json) file")
    args = parser.parse_args()
    CFG = None
    if args.config is not None:
        s = importlib.util.spec_from_file_location("", os.path.abspath(args.config))
        confmod = importlib.util.module_from_spec(s)
        s.loader.exec_module(confmod)
        CFG = dict(confmod.CFG)

    train(*build_config(CFG))
