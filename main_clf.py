"""Train the classifier head of the unsupervised model."""
import torch
import torch.nn as nn
from networks.unsupervised import ReceptiveFieldClassifier
from datasets.unsupervised import _load_mnist

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ReceptiveFieldClassifier().to(DEVICE)

pretrained_dict = torch.load("models/test.pt")
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

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)

# Load data (mnist)
train, test = _load_mnist("data/")
train_loader = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=64, shuffle=True, num_workers=0
)

# Logging
writer = SummaryWriter("runs/classifier")


# Training loop for the classifier head
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

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
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
    writer.add_scalar("Loss/Train_epoch", epoch_loss, epoch)

    # Validation loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

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
