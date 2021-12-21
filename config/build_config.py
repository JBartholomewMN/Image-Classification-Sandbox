from dataset.coco_dataset import CocoDetection
from dataset.imagenet import ImageNetClassifcation
from config.default import CFG


def build_config(config=CFG):

    # model
    config["model"] = config["model"](config)

    for m in config["model_inits"]:
        m(config["model"])

    config["model"] = config["model"].to(config["device"])
    print(
        "trainable params: ",
        sum(p.numel() for p in config["model"].parameters() if p.requires_grad),
    )

    # optimizer
    config["optimizer"] = config["optimizer"](
        config["model"].parameters(), **config["optimizer_args"]
    )

    # train dataloader
    train_dataset = config["dataset"](
        atransforms=config["A_transforms"],
        transform=config["T_transforms"],
        **config["trainset_args"]
    )
    config["trainloader"] = config["trainloader"](
        train_dataset, **config["trainloader_args"]
    )

    # test dataloader
    test_dataset = config["dataset"](
        atransforms=config["A_transforms_test"],
        transform=config["T_transforms"],
        **config["testset_args"]
    )
    config["testloader"] = config["testloader"](
        test_dataset, **config["testloader_args"]
    )

    return (
        config["model"],
        config["optimizer"],
        config["criterion"],
        config["trainloader"],
        config["testloader"],
        config["epochs"],
        config["device"],
        config,
    )
