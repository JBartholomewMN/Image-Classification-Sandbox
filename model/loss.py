import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, lclass=1, lnoobj=10, lobj=1, lbox=10):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lclass = lclass
        self.lnoobj = lnoobj
        self.lobj = lobj
        self.lbox = lbox

    def forward(self, preds, target, anchors):

        # extract bounding boxes for each image object detection
        gt_bbs = list()
        for img in target:
            gt_bbs.append(torch.tensor([det["bbox"] for det in img]).to(preds.device))

        # object loss

        # no object detection loss

        # box coordinate loss

        # class loss

        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0