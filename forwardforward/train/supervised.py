import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader


def train_supervised(
    model: torch.nn.Module,
    positive_loader: DataLoader,
    negative_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    writer: torch.utils.tensorboard.SummaryWriter,
):
    best_accuracy = 0

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    running_batch_idx = 0
    running_total_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train pass
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
                for layer_idx, layer_goodness in enumerate(
                    layer_goodnesses_pos
                ):
                    writer.add_scalar(
                        f"Layer {layer_idx} Pos Goodness",
                        layer_goodness,
                        running_batch_idx,
                    )

                for layer_idx, layer_goodness in enumerate(
                    layer_goodnesses_neg
                ):
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

        # Test pass
        if epoch % 5 == 0:
            model.eval()

            progress_bar = tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"Test",
                unit="batch",
            )

            correct = 0
            total = 0

            for batch_idx, test_batch in progress_bar:
                images, labels = test_batch
                images = images.to(device)
                labels = labels.to(device)

                predicted_labels = model.forward_supervised(images)  # [B]

                # Calculate accuracy
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy}%")

            if writer is not None:
                writer.add_scalar(
                    f"Test Accuracy",
                    accuracy,
                    running_batch_idx,
                )

            # Save the model if it's the best so far
            if accuracy > best_accuracy:
                torch.save(
                    model.state_dict(),
                    f"models/supervised_{timestamp}_epoch-{epoch}_acc-{accuracy}.pt",
                )
                best_accuracy = accuracy

    writer.close()
    print(
        f"Finished training of supervised network at {datetime.now().strftime('%H:%M:%S')}"
    )
