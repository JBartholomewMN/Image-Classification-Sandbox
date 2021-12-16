from numpy import histogram_bin_edges
import torch
import torch.nn as nn
from utils.utils import iou_width_height as iouwh
from utils.utils import intersection_over_union as iou


class YoloLoss(nn.Module):
    def __init__(self, lclass=1.0, lnoobj=10.0, lobj=1.0, lbox=10.0):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lclass = torch.tensor(lclass).float()
        self.lnoobj = torch.tensor(lnoobj).float()
        self.lobj = torch.tensor(lobj).float()
        self.lbox = torch.tensor(lbox).float()

    def forward(self, X, yboxes, ylabels, anchors, cfg):

        anchors = torch.tensor(anchors)
        loss = None
        hgrid = X.shape[-3]
        wgrid = X.shape[-4]

        # build a mask thats (nanchors, h, w)
        taken = torch.zeros((anchors.shape[0], hgrid, wgrid))
        loss = torch.zeros(1).float().to(X.device)

        # determine which anchor boxes are responsible for making a prediction
        for labs, boxes, pred in zip(ylabels, yboxes, X):
            for box, lab in zip(boxes, labs):

                # find which anchor box this object belongs to
                y = torch.floor(box[1] * hgrid).int()
                x = torch.floor(box[0] * wgrid).int()

                # find the center coordinate relative to the anchor box corner
                # for example, (xbox - xanchor_corner) / cell_width
                yrelanchor = (box[1] - (y / hgrid)) * hgrid
                xrelanchor = (box[0] - (x / wgrid)) * wgrid

                # find the iou of each anchor box at this scale
                iou_anchors = iouwh(box[2:4].detach(), anchors)

                for anch in iou_anchors.argsort(descending=True, dim=0):
                    # if anchor isn't used, claim it and update the loss
                    if not torch.any(taken[anch, y, x]):
                        taken[anch, y, x] = 1

                        # objectness loss (target: 1)
                        loss += self.lobj * self.bce(
                            pred[anch, y, x, 0:1], torch.tensor([1.0]).to(pred.device)
                        )

                        # compute the coeffs that the network should learn
                        # which scale the predefined bounding boxes aka 'the prior'
                        w_coeff = box[2] / anchors[anch][0]
                        h_coeff = box[3] / anchors[anch][1]

                        # coordinate loss
                        loss += self.lbox * self.mse(
                            pred[anch, y, x, 1:5],
                            torch.tensor([xrelanchor, yrelanchor, w_coeff, h_coeff]).to(
                                pred.device
                            ),
                        )

                        # classification loss
                        loss += self.lclass * self.entropy(
                            torch.unsqueeze(pred[anch, y, x, 5:], dim=0),
                            torch.unsqueeze(
                                nn.functional.one_hot(lab, cfg["nclasses"])
                                .float()
                                .to(pred.device),
                                dim=0,
                            ),
                        )

            # now loop over all grid locations that aren't supposed to be an object
            # and determine if they should be punished
            for a, gh, gw in (taken == 0).nonzero(as_tuple=False):
                pbox = pred[anch, gh, gw, 0:5]
                # scale the predefined anchor box
                sbox = pbox.clone()
                sbox[3] = pbox[3] * anchors[a][0]
                sbox[4] = pbox[4] * anchors[a][1]

                objectness = pred[anch, gh, gw, 0]

                print(a, gh, gw)

            # keep the best anchor, determine if it's
            iou_anchors[iou_anchors > cfg["iou_thresh"]]

            print(iou_anchors)

        # no object detection loss

        # box coordinate loss

        # class loss

        # obj = target[..., 0] == 1
        # noobj = target[..., 0] == 0
