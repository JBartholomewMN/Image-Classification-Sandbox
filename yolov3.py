from os import pipe2
import torch
import torch.nn as nn
import yaml

class ResBlock(nn.Module):
    def __init__(self, inchans:list, outchans:list, ksizes:list, repeats:int, store:bool):
        super(ResBlock, self).__init__()
        layers = list()
        
        # build the layers and keep track of
        # where the skip connections need to happen
        self.skipind = list()
        for _ in range(repeats):
            for ic, oc, ks in zip(inchans, outchans, ksizes):
                layers += [
                    ConvBlock(ic, oc, ks, padding='same')
                ]
            self.skipind += [len(layers) - 1]

        self.layers = nn.ModuleList(layers)
        self.store = store
        

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
            ConvBlock(inchans, 2 * inchans, ksize=3, padding=1),
            ConvBlock(
                2 * inchans, 
                (nclasses + nbbvals) * nanchors, 
                bnorm=False, 
                ksize=1
            )
        )
        self.nclasses = nclasses
        self.nanchors = nanchors
        self.bboxvals = nbbvals


    def forward(self, x):
        # return the predictions in the form (batch, nanchors, h, w, nclasses+bboxvals)
        return (
            self.layers(x).reshape(
                x.shape[0], self.nanchors, self.nclasses + self.bboxvals, x.shape[2], x.shape[3]
            ).permute(
                0, 1, 3, 4, 2
            )
        )

class ConvBlock(nn.Module):
    def __init__(
        self, 
        inchans, 
        outchans, 
        ksize, 
        stride=1, 
        padding='same', 
        bnorm=True
    ):
        super(ConvBlock, self).__init__()
        layers = list()
        layers.append(
            nn.Conv2d(inchans, outchans, ksize, stride, padding, bias=not bnorm)
        )
        
        if bnorm:
            layers.append(nn.BatchNorm2d(outchans))
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.ModuleList(layers)

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
                    ConvBlock(inchans, outchans, ksize, stride, padding='same' if stride==1 else stride // 2)
                ]
            elif v[0] == "res":
                _, inchans, outchans, ksize, repeats, store = v
                layers += [
                    ResBlock(inchans, outchans, ksize, repeats, store)
                ]

        self.layers = nn.ModuleList(layers)

        if weights is not None:
            pass

    def forward(self, x):
        saved_outputs = list()
        for l in self.layers:
            x = l(x)
            if hasattr(l, 'store') and l.store:
                saved_outputs += [x] 

        return saved_outputs

class YoloV3(nn.Module):
    def __init__(self, config="config.yaml"):
        super(YoloV3, self).__init__()

        # load config data from yaml
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        self.backbone = Backbone(cfg)

        # build the predictors for each res layer where the output is True
        # in the config file...
        predictors = list()
        nchans = 0
        for l in reversed(list(filter(lambda x: x[0] == "res" and x[-1], cfg["layers"].values()))):
            _, _, outchans, _, _, _ = l
            nchans += outchans[-1]
            predictors += [ScalePrediction(nchans, cfg["nclasses"], cfg["nanchors"], cfg["nbbvals"])]
        self.predictors = nn.ModuleList(predictors)

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
        for ext, pred in zip(extractions, self.predictors):
            
            if catted is not None:
                catted = torch.cat((upsample(catted), ext), dim=1)
            else:
                catted = ext

            outputs.append(pred(catted))

        return outputs


if __name__ == "__main__":
    # a little bit of test code
    model = YoloV3()
    model.to('cuda')
    test_data = torch.randn(10, 3, 416, 416).to('cuda')
    d = model.forward(test_data)
    for out in d:
        print('output_shape:', out.shape)