from os import pipe2
import torch
import torch.nn as nn
import math


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pe


class ConvFormer(nn.Module):
    def __init__(
        self,
        inchans: list,
        kqvchans: list,
        nheads: list,
        hiddensize: int,
        outchans: int,
        repeats: int,
        store: bool,
    ):
        super(ConvFormer, self).__init__()
        layers = list()

        self.ks = nn.ModuleList(
            [
                nn.Conv2d(inchans, kqvchans, kernel_size=1, padding="same")
                for _ in range(nheads)
            ]
        )
        self.qs = nn.ModuleList(
            [
                nn.Conv2d(inchans, kqvchans, kernel_size=1, padding="same")
                for _ in range(nheads)
            ]
        )
        self.vs = nn.ModuleList(
            [
                nn.Conv2d(inchans, kqvchans, kernel_size=1, padding="same")
                for _ in range(nheads)
            ]
        )
        self.sm = nn.Softmax2d()

        self.ll = nn.Linear(nheads * kqvchans, inchans)
        self.leaky = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm2d(inchans)
        self.inchans = inchans
        self.outchans = outchans

    def forward(self, x):

        pos_encoding = positionalencoding2d(self.inchans, x.shape[-2], x.shape[-1]).to(
            x.device
        )
        x = x + pos_encoding
        ks = (
            torch.stack([k(x) for k in self.ks])
            .flatten(start_dim=-2)
            .permute(1, 0, 3, 2)
        )
        qs = (
            torch.stack([q(x) for q in self.qs])
            .flatten(start_dim=-2)
            .permute(1, 0, 2, 3)
        )
        vs = (
            torch.stack([v(x) for v in self.vs])
            .flatten(start_dim=-2)
            .permute(1, 0, 3, 2)
        )

        kqt = torch.matmul(ks, qs) / ks.shape[-2]
        sm = self.sm(kqt)
        att = torch.matmul(sm, vs).permute(0, 2, 3, 1)
        att = att.flatten(start_dim=-2)
        projed = self.leaky(self.ll(att)).permute(0, 2, 1)
        projed = projed.reshape(x.shape)

        return self.bn(projed + x)


class ResBlock(nn.Module):
    def __init__(
        self,
        inchans: list,
        outchans: list,
        ksizes: list,
        repeats: int,
        store: bool,
        use_residual: bool = True,
    ):
        super(ResBlock, self).__init__()
        layers = list()

        # build the layers and keep track of
        # where the skip connections need to happen
        self.skipind = list()
        for _ in range(repeats):
            for ic, oc, ks in zip(inchans, outchans, ksizes):
                layers += [ConvBlock(ic, oc, ks, padding="same")]
            if use_residual:
                self.skipind += [len(layers) - 1]

        self.layers = nn.ModuleList(layers)
        self.store = store
        self.outchans = outchans[-1]

    def forward(self, x):

        # loop over the layers and execute the skip
        # connections at the proper places
        origx = x
        for i, l in enumerate(self.layers):
            if i in self.skipind:
                x = l(x) + origx
                origx = x
            else:
                x = l(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, inchans, nclasses, nanchors, nbbvals):
        super(ScalePrediction, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(inchans, inchans // 2, ksize=3),
            ConvBlock(
                inchans // 2, (nclasses + nbbvals) * nanchors, bnorm=False, ksize=1
            ),
        )
        self.nclasses = nclasses
        self.nanchors = nanchors
        self.bboxvals = nbbvals

    def forward(self, x):
        # return the predictions in the form (batch, nanchors, h, w, nclasses+bboxvals)
        return (
            self.layers(x)
            .reshape(
                x.shape[0],
                self.nanchors,
                self.nclasses + self.bboxvals,
                x.shape[2],
                x.shape[3],
            )
            .permute(0, 1, 3, 4, 2)
        )


class ConvBlock(nn.Module):
    def __init__(self, inchans, outchans, ksize, stride=1, padding="same", bnorm=True):
        super(ConvBlock, self).__init__()
        layers = list()
        layers.append(
            nn.Conv2d(inchans, outchans, ksize, stride, padding, bias=not bnorm)
        )

        if bnorm:
            layers.append(nn.BatchNorm2d(outchans))
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.ModuleList(layers)
        self.outchans = outchans

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Backbone(nn.Module):
    def __init__(self, cfg, weights=None):
        super(Backbone, self).__init__()

        # build the backbone programatically
        layers = list()
        for _, v in cfg["layers"].items():
            if v[0] == "conv":
                _, inchans, outchans, ksize, stride = v
                layers += [
                    ConvBlock(
                        inchans,
                        outchans,
                        ksize,
                        stride,
                        padding="same" if stride == 1 else stride // 2,
                    )
                ]
            elif v[0] == "res":
                _, inchans, outchans, ksize, repeats, store = v
                layers += [ResBlock(inchans, outchans, ksize, repeats, store)]
            elif v[0] == "convformer":
                # ["convtrans", inchans, kqvchans, nheads, hiddensize, repeats, store_output]
                _, inchans, kqvchans, nheads, hiddensize, outchans, repeats, store = v
                layers += [
                    ConvFormer(
                        inchans, kqvchans, nheads, hiddensize, outchans, repeats, store
                    )
                ]

        self.layers = nn.ModuleList(layers)

        if weights is not None:
            pass

        # used if in classifier mode:
        self.classifier = False
        self.nclasses = cfg["nclasses"]

    def forward(self, x):
        saved_outputs = list()
        for l in self.layers:
            x = l(x)
            if hasattr(l, "store") and l.store:
                saved_outputs += [x]

        if not self.classifier:
            return
        else:
            x = self.conv_class_layer(x)
            x = self.rl(x)
            x = torch.mean(x, (-1, -2))
            x = self.fc(x)
            if not self.training:
                x = self.sm(x)
            return x

    def add_classifier(cls):
        cls.classifier = True
        cls.conv_class_layer = nn.Conv2d(
            cls.layers[-1].outchans, cls.nclasses, 1, padding="same"
        )
        cls.rl = nn.LeakyReLU()
        cls.fc = nn.Linear(cls.nclasses, cls.nclasses)
        cls.sm = nn.Softmax(-1)


class YoloV3(nn.Module):
    def __init__(self, cfg):
        super(YoloV3, self).__init__()

        self.backbone = Backbone(cfg)
        self.anchors = torch.tensor(cfg["anchors"]).to(cfg["device"])
        # build the predictors at different scales.
        # uses the boolean in res layers from the config file...
        predictors = list()
        nchans = 0
        for l in reversed(
            list(filter(lambda x: x[0] == "res" and x[-1], cfg["layers"].values()))
        ):
            _, _, outchans, _, _, _ = l
            nchans += outchans[-1]
            predictors += [
                ScalePrediction(
                    nchans, cfg["nclasses"], cfg["nanchors"], cfg["nbbvals"]
                )
            ]
            self.predictors = nn.ModuleList(predictors)
        print("total parameters: ", sum(p.numel() for p in self.parameters()))
        print(
            "trainanble parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(self, x):
        # get the feature maps from the backbone
        # arrange them from coarsest spacial to finest spatial
        # aka deepest to shallowest
        extractions = reversed(self.backbone(x))

        # put each feature map through a scaled predictor
        # this is where the upsampling happens for now
        outputs = list()
        upsample = nn.Upsample(scale_factor=2)
        catted = None
        for ext, pred, anch in zip(extractions, self.predictors, self.anchors):

            if catted is not None:
                catted = torch.cat((upsample(catted), ext), dim=1)
            else:
                catted = ext

            x = pred(catted)

            if not self.training:
                # sigmoid the x, y (relative to anchor box corner) bounding them from 0 to 1
                # then move them out of grid space
                ny, nx = x.shape[2:4]
                grid = self._make_grid(nx, ny).to(x.device)
                x[..., 0:2] = torch.sigmoid(x[..., 0:2]) + grid
                x[..., 0] /= nx
                x[..., 1] /= ny

                # sigmoid the objectness score [0, 1]
                x[..., 4] = torch.sigmoid(x[..., 4])

                # exponeniate the h, w scale
                # then apply the scaling to the anchor boxs
                x[..., 2:4] = (torch.exp(x[..., 2:4]).movedim(1, -2) * anch).movedim(
                    -2, 1
                )
                x = x.reshape(x.shape[0], -1, x.shape[-1])

            outputs.append(x)

        return torch.cat(outputs, 1) if not self.training else outputs

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


if __name__ == "__main__":
    # a little bit of test code
    model = YoloV3()
    print(model)
    model.to("cuda")
    test_data = torch.randn(10, 3, 416, 416).to("cuda")
    d = model.forward(test_data)
    for out in d:
        print("output_shape:", out.shape)
