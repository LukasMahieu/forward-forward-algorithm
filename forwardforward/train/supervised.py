import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader


def train_supervised(
    model: torch.nn.Module,
    positive_loader: DataLoader,
    negative_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    writer: torch.utils.tensorboard.SummaryWriter,
):
    lowest_total_loss = 100

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    running_batch_idx = 0
    running_total_loss = 0.0

    for epoch in range(1, num_epochs + 1):
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
            positive_images = positive_images.to(device)
            negative_images = negative_images.to(device)

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
            if writer is not None:
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

            if writer is not None:
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
            average_total_loss = np.round(running_total_loss / running_batch_idx, 3)
            if average_total_loss < lowest_total_loss:
                torch.save(
                    model.state_dict(),
                    f"models/unsupervised_{timestamp}_{epoch}_{average_total_loss}.pt",
                )

    writer.close()
    print(
        f"Finished training of supervised network at {datetime.now().strftime('%H:%M:%S')}"
    )
