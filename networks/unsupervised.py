import torch.nn as nn
import torch.optim as optim
import torch


"""Receptive Field Network for unsupervised learning."""


class ReceptiveFieldNet(nn.Module):
    def __init__(self, device):
        super(ReceptiveFieldNet, self).__init__()
        layer1 = ReceptiveFieldLayer(1, 128, 10, 6).to(device)
        layer2 = ReceptiveFieldLayer(128, 220, 2, 1).to(device)
        layer3 = ReceptiveFieldLayer(220, 512, 2, 1).to(device)
        self.layers = [layer1, layer2, layer3]

    def train_batch(self, x, datatype: str):
        h = x
        layer_losses = []
        layer_goodnesses = []
        for i, layer in enumerate(self.layers):
            h, loss, goodness = layer.train(h, datatype)
            layer_losses.append(loss)
            layer_goodnesses.append(goodness)
        return layer_losses, layer_goodnesses


class ReceptiveFieldLayer(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, lr=0.001):
        super(ReceptiveFieldLayer, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.threshold = 50  # Arbitrary threshold > 0 (pref > 2 imo)

    def forward(self, x, epsilon=1e-8):
        x_normalized = x / (epsilon + x.norm(dim=1, keepdim=True))
        output = self.relu(self.conv(x_normalized))
        return output

    def train(self, x, datatype: str, epsilon=1e-8):
        """Train the layer for one batch of positive or negative data."""
        self.opt.zero_grad()

        goodness = torch.sum(torch.square(self.forward(x)))  # [0;inf]

        if datatype == "pos":
            loss = -torch.log(goodness + epsilon)
        elif datatype == "neg":
            loss = torch.log(goodness + epsilon)
        loss.backward()
        self.opt.step()

        return self.forward(x).detach(), loss.item(), goodness.item()


class ReceptiveFieldClassifier(nn.Module):
    def __init__(self, device):
        super(ReceptiveFieldClassifier, self).__init__()
        self.layer1 = nn.ReLU(nn.Conv2d(1, 128, 10, 6))
        self.layer2 = nn.ReLU(nn.Conv2d(128, 220, 2, 1))
        self.layer3 = nn.ReLU(nn.Conv2d(220, 512, 2, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        h = self.fc(h)
        return h


### Fully connected not implemented yet ###


class FullyConnected(nn.Module):
    """Fully connected network for the unsupervised learning task. Has four hidden layer of 2000 ReLU units each."""

    def __init__(self, dim_input=28 * 28 * 1):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(dim_input, 2000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.
        Note: don't forget to apply layer normalization to the output of each layer.
        """
        output1 = self.relu1(self.fc1(x))
        output2 = self.relu2(self.fc2(output1))
        output3 = self.relu3(self.fc3(output2))
        output4 = self.relu4(self.fc4(output3))
        return output1, output2, output3, output4
