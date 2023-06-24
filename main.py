"""Test forward-forward algorithm (Hinton) on MNIST using unsupervised and supervised learning."""

import os
import argparse
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from forwardforward.datasets.unsupervised import (
    create_mnist_datasets_unsupervised,
    _load_mnist,
)
from forwardforward.datasets.supervised import create_mnist_datasets_supervised

from forwardforward.train.unsupervised_backbone import train_unsupervised_backbone
from forwardforward.train.unsupervised_head import train_unsupervised_clf
from forwardforward.train.supervised import train_supervised

from forwardforward.networks.unsupervised import (
    ReceptiveFieldNet,
    ReceptiveFieldClassifier,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_unsupervised_backbone", action="store_true")
    parser.add_argument("--train_unsupervised_clf", action="store_true")
    parser.add_argument("--train_supervised", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--no_logs", default=False, action="store_true", help="Don't log to tensorboard"
    )
    parser.add_argument(
        "--pretrained_backbone_filename",
        type=str,
        default="",
        help="Optional filename of pretrained backbone in /models. Will use latest backbone in models/ if not specified.",
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #### unsupervised backbone ####
    if args.train_unsupervised_backbone:
        model = ReceptiveFieldNet(device).to(device)

        positive, negative, _ = create_mnist_datasets_unsupervised("data")

        positive_loader = DataLoader(
            positive, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        negative_loader = DataLoader(
            negative, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        if not args.no_logs:
            writer = SummaryWriter(f"runs/unsupervised_backbone")
        else:
            writer = None

        train_unsupervised_backbone(
            model, positive_loader, negative_loader, device, args.num_epochs, writer
        )

    #### unsupervised classifier head ####
    if args.train_unsupervised_clf:
        model = ReceptiveFieldClassifier().to(device)
        if args.pretrained_backbone_filename != "":
            pretrained_backbone_path = os.path.join(
                "models", args.pretrained_backbone_filename
            )
        else:
            pretrained_backbone_path = os.path.join("models", os.listdir("models")[-1])
        print(f"Loading pretrained backbone from {pretrained_backbone_path}")

        # Load pretrained backbone
        pretrained_dict = torch.load(pretrained_backbone_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # Freeze the convolutional layers
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False

        # Load data (mnist)
        train, test = _load_mnist("data/")
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=64, shuffle=True, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=64, shuffle=True, num_workers=0
        )

        if not args.no_logs:
            writer = SummaryWriter(f"runs/unsupervised_head")
        else:
            writer = None

        train_unsupervised_clf(
            model, train_loader, test_loader, device, args.num_epochs, writer
        )

    #### Supervised network ####
    if args.train_supervised:
        model = ReceptiveFieldNet(device).to(device)

        positive, negative, _ = create_mnist_datasets_supervised("data")

        positive_loader = DataLoader(
            positive, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        negative_loader = DataLoader(
            negative, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        if not args.no_logs:
            writer = SummaryWriter(f"runs/supervised")
        else:
            writer = None

        train_supervised(
            model, positive_loader, negative_loader, device, args.num_epochs, writer
        )


if __name__ == "__main__":
    print("Starting at:", datetime.now().strftime("%Y-%m-%d_%Hh%M"))
    print("-" * 80)

    args = parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("-" * 80)

    main(args)

    print("Finished at:", datetime.now().strftime("%Y-%m-%d_%Hh%M"))
