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
    top5accbest = 0
    t1metric = torchmetrics.Accuracy()
    t5metric = torchmetrics.Accuracy(top_k=5)
    running_loss = 0.0

    for epoch in range(epochs):
        model.train()
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

        model.eval()
        for timgb, tlabs in tqdm.tqdm(testloader, desc="Testing"):
            with torch.no_grad():
                preds = model(timgb.to(device)).detach().cpu()
                acc1 = t1metric(preds, tlabs)
                acc5 = t5metric(preds, tlabs)

        acc1 = t1metric.compute()
        acc5 = t5metric.compute()

        if acc1 > top1accbest:
            top1accbest = acc1
            top5accbest = acc5
            torch.save(model.state_dict(), cfg["weights_save_path"])

        print("Epoch Top-1 accuracy: ", acc1)
        print("Epoch Top-5 accuracy: ", acc5)
        print("Best  Top-1 accuracy: ", top1accbest)
        print("Best  Top-5 accuracy: ", top5accbest)


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
