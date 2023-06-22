import os

import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime

from networks.unsupervised import ReceptiveFieldNet
from datasets.unsupervised import create_mnist_datasets

# Hyperparams
BATCH_SIZE = 1000  # So 60 batches since 60k images
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Create a dataset & dataloader (MNIST)
positive, negative, test = create_mnist_datasets("data")
model = ReceptiveFieldNet(DEVICE).to(DEVICE)

# Create data loaders
positive_loader = DataLoader(
    positive, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
negative_loader = DataLoader(
    negative, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# Logging
writer = SummaryWriter("runs/unsupervised")
lowest_total_loss = 100
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

running_batch_idx = 0
running_total_loss = 0.0

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()

    progress_bar = tqdm(
        enumerate(zip(positive_loader, negative_loader)),
        total=min(len(positive_loader), len(negative_loader)),
        desc=f"Epoch {epoch}",
        unit="batch",
    )

    for batch_idx, (positive_batch, negative_batch) in progress_bar:
        batch_layer_losses = {}
        for i, layer in enumerate(model.layers):
            batch_layer_losses[i] = 0.0

        running_batch_idx += 1

        positive_images, _ = positive_batch
        negative_images, _ = negative_batch
        positive_images = positive_images.to(DEVICE)
        negative_images = negative_images.to(DEVICE)

        # Positive pass
        layer_losses, layer_goodnesses_pos = model.train_batch(
            positive_images, "pos"
        )

        for i, layer_loss in enumerate(layer_losses):
            batch_layer_losses[i] += layer_loss

        # Negative pass
        layer_losses, layer_goodnesses_neg = model.train_batch(
            negative_images, "neg"
        )

        for i, layer_loss in enumerate(layer_losses):
            batch_layer_losses[i] += layer_loss

        # Logging
        for layer_idx, layer_goodness in enumerate(layer_goodnesses_pos):
            writer.add_scalar(
                f"Layer {layer_idx} Pos Goodness",
                layer_goodness,
                running_batch_idx,
            )

        for layer_idx, layer_goodness in enumerate(layer_goodnesses_neg):
            writer.add_scalar(
                f"Layer {layer_idx} Neg Goodness",
                layer_goodness,
                running_batch_idx,
            )

        # Print progress every 2 batches
        batch_total_loss = 0.0
        for layer_idx, batch_layer_loss in batch_layer_losses.items():
            writer.add_scalar(
                f"Layer {layer_idx} Batch Loss",
                batch_layer_loss,
                running_batch_idx,
            )
            batch_total_loss += batch_layer_loss

        writer.add_scalar(
            f"Total Batch Loss",
            batch_total_loss,
            running_batch_idx,
        )
        postfix = {
            f"Layer {layer_idx}": batch_layer_loss
            for layer_idx, batch_layer_loss in batch_layer_losses.items()
        }
        progress_bar.set_postfix(postfix)

        running_total_loss += batch_total_loss

    # Save the model every 5 epochs if it's the best so far
    if epoch % 5 == 0:
        average_total_loss = np.round(
            running_total_loss / running_batch_idx, 3
        )
        if average_total_loss < lowest_total_loss:
            torch.save(
                model.state_dict(),
                f"models/unsupervised_{timestamp}_{epoch}_{average_total_loss}.pt",
            )

writer.close()
print(f"Finished training at {datetime.now().strftime('%H:%M:%S')}")
