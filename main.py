import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datetime import datetime

from networks.unsupervised import ReceptiveFieldNet
from datasets.unsupervised import create_mnist_datasets

# Hyperparams
BATCH_SIZE = 1024
NUM_EPOCHS = 100
THRESHOLD = 1  # Arbitrary threshold > 0
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

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()

    running_layer_losses = {}
    for i, layer in enumerate(model.layers):
        running_layer_losses[i] = 0.0

    progress_bar = tqdm(
        enumerate(zip(positive_loader, negative_loader)),
        total=min(len(positive_loader), len(negative_loader)),
        desc=f"Epoch {epoch}",
        unit="batch",
    )

    for batch_idx, (positive_batch, negative_batch) in progress_bar:
        positive_images, _ = positive_batch
        negative_images, _ = negative_batch
        positive_images = positive_images.to(DEVICE)
        negative_images = negative_images.to(DEVICE)

        # Positive pass
        layer_losses, layer_goodnesses_pos = model.train_batch(positive_images, "pos")

        for i, layer_loss in enumerate(layer_losses):
            running_layer_losses[i] += layer_loss

        # Negative pass
        layer_losses, layer_goodnesses_neg = model.train_batch(negative_images, "neg")

        for i, layer_loss in enumerate(layer_losses):
            running_layer_losses[i] += layer_loss

        # Logging
        for layer_idx, layer_goodness in enumerate(layer_goodnesses_pos):
            writer.add_scalar(
                f"Layer {layer_idx} Pos Goodness",
                layer_goodness,
                (epoch - 1) * len(positive_loader) + batch_idx,
            )

        for layer_idx, layer_goodness in enumerate(layer_goodnesses_neg):
            writer.add_scalar(
                f"Layer {layer_idx} Neg Goodness",
                layer_goodness,
                (epoch - 1) * len(positive_loader) + batch_idx,
            )

        # Print progress every 2 batches
        if batch_idx % 2 == 0:
            model_params = model.state_dict()
            # test saving
            torch.save(model_params, "models/test.pt")

            total_loss = 0.0
            for layer_idx, running_loss in running_layer_losses.items():
                average_loss = running_loss / (batch_idx * 2 + 1)
                writer.add_scalar(
                    f"Layer {layer_idx} Average Batch Loss",
                    average_loss,
                    (epoch - 1) * len(positive_loader) + batch_idx,
                )
                total_loss += average_loss
            writer.add_scalar(
                f"Total Average Batch Loss",
                total_loss,
                (epoch - 1) * len(positive_loader) + batch_idx,
            )
            postfix = {
                f"Layer {layer_idx}:": running_loss / (10 * 2)
                for layer_idx, running_loss in running_layer_losses.items()
            }
            progress_bar.set_postfix(postfix)

    # Calculate average loss for each layer every 5 epochs
    # and save the model if the average loss accross layers is the lowest so far
    if epoch % 5 == 0:
        total_loss = 0.0
        for layer_idx, running_loss in running_layer_losses.items():
            average_loss = running_loss / (len(positive_loader) + len(negative_loader))

            print(f"Epoch: {epoch}, Average Layer {layer_idx} Loss: {average_loss:.4f}")
            total_loss += average_loss
        if total_loss < lowest_total_loss:
            lowest_total_loss = total_loss
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(
                model.state_dict(),
                f"models/unsupervised_{timestamp}_{epoch}.pt",
            )

writer.close()
