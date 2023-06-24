"""Receptive Field Network for unsupervised learning."""
import torch.nn as nn
import torch.optim as optim
import torch

from collections import OrderedDict


def _overlay_y_on_x_batch(images: torch.Tensor, label: int):
    """Create a positive sample by taking an image and replacing the first 10 pixels by a one of N representation of the label"""
    overlayed_images = images.clone()
    for i in range(images.shape[0]):
        overlayed_images[i, 0, 0, :10] = overlayed_images.min()
        overlayed_images[i, 0, 0, label] = overlayed_images.max()
    return overlayed_images


class ReceptiveFieldNet(nn.Module):
    def __init__(self, device):
        super(ReceptiveFieldNet, self).__init__()
        layer1 = ReceptiveFieldLayer(1, 128, 10, 6).to(device)
        layer2 = ReceptiveFieldLayer(128, 220, 2, 1).to(device)
        layer3 = ReceptiveFieldLayer(220, 512, 2, 1).to(device)

        self.layers = [layer1, layer2, layer3]

    def train_batch(self, x, datatype: str):
        """"""
        h = x
        layer_losses = []
        layer_goodnesses = []
        for i, layer in enumerate(self.layers):
            h, loss, goodness = layer.train(h, datatype)
            layer_losses.append(loss)
            layer_goodnesses.append(goodness)
        return layer_losses, layer_goodnesses

    def state_dict(self):
        """Return state dict with layer names"""
        state = OrderedDict()
        for i, layer in enumerate(self.layers):
            layer_state = layer.state_dict()
            for name, param in layer_state.items():
                name = name.split(".")[1]  # Remove "conv" from name
                new_name = f"layer{i+1}.{name}"
                state[new_name] = param
        return state

    def load_weights_dict(self, state_dict):
        """Load state dict for each layer."""
        iter_state_dict = iter(state_dict.items())
        for weight, bias in zip(iter_state_dict, iter_state_dict):
            layer_idx = int(weight[0][5]) - 1
            layer = self.layers[layer_idx]
            combined_dict = {
                "conv.weight": weight[1],
                "conv.bias": bias[1],
            }

            layer.load_state_dict(combined_dict)

    def forward_supervised(self, x):
        """Forward pass for supervised learning.

        Note: Don't use this for unsupervised learning (use ReceptiveFieldClassifier instead).
        """
        goodness_per_label = []
        for label in range(10):
            h = _overlay_y_on_x_batch(x, label)
            layer_goodness = []
            for layer in self.layers:
                h = layer(h)  # [B, C, H, W]
                layer_goodness.append(
                    torch.mean(torch.square(h), dim=(1, 2, 3))
                )  # [B]
            goodness_per_label.append(
                torch.sum(torch.stack(layer_goodness, dim=1), dim=1)
            )
        argmax = torch.stack(goodness_per_label, dim=1).argmax(1)  # [B]
        return argmax


class ReceptiveFieldLayer(nn.Module):
    """Individual layer of ReceptiveFieldNet."""

    def __init__(
        self, C_in: int, C_out: int, kernel_size: int, stride: int, lr=0.001
    ):
        super(ReceptiveFieldLayer, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.threshold = 2  # Arbitrary threshold > 0 (pref > 2 imo)

    def forward(self, x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        x_normalized = x / (epsilon + x.norm(dim=1, keepdim=True))
        output = self.relu(self.conv(x_normalized))
        return output

    def train(self, x: torch.Tensor, datatype: str, epsilon=1e-8) -> tuple:
        """Train the layer for one batch of positive or negative data."""
        self.opt.zero_grad()

        goodness = torch.mean(
            torch.square(self.forward(x))
        )  # [0;inf], accross batch

        if datatype == "pos":
            loss = -torch.sigmoid(goodness - self.threshold)
        elif datatype == "neg":
            loss = torch.sigmoid(goodness - self.threshold)

        loss.backward()
        self.opt.step()

        return self.forward(x).detach(), loss.item(), goodness.item()


class ReceptiveFieldClassifier(nn.Module):
    """Classifier for ReceptiveFieldNet.
    Use to train classifier head on top of Unsupervised ReceptiveFieldNet."""

    def __init__(self):
        super(ReceptiveFieldClassifier, self).__init__()
        self.layer1 = nn.Conv2d(1, 128, 10, 6)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Conv2d(128, 220, 2, 1)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Conv2d(220, 512, 2, 1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        h = self.layer1(x)
        h = self.relu1(h)

        h = self.layer2(h)
        h = self.relu2(h)

        h = self.layer3(h)
        h = self.relu3(h)

        h = self.flatten(h)
        h = self.fc(h)
        return h


if __name__ == "__main__":
    # Test load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReceptiveFieldNet(device)
    model.load_weights_dict(
        torch.load("models/supervised_2023-06-24_16-54-37_5_8.45.pt")
    )
    # Check that each layer has loaded weights
    for i, layer in enumerate(model.layers):
        print(f"layer {i+1}")
        print(layer.conv.weight.shape)
        print(layer.conv.bias.shape)
