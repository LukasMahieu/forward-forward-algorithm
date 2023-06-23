"""Train the classifier head of the unsupervised model."""
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime


def train_unsupervised_clf(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
    writer: torch.utils.tensorboard.SummaryWriter,
):
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        # Training loop
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Tensorboard logging
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train", loss.item(), step)

            progress_bar.set_postfix({"Loss": loss.item()})

        # Print the epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar("Loss/Train_epoch", epoch_loss, epoch)

        # Validation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        print(f"Validation Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        writer.add_scalar("Loss/Validation", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

    # Close the Tensorboard writer
    writer.close()
    print(
        f"Finished training of unsupervised clf head at {datetime.now().strftime('%H:%M:%S')}"
    )
