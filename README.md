## Installation

```
python -m venv .venv
source .venv\bin\activate # or .venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
```

Install latest version of [pytorch and torchvision](https://pytorch.org/get-started/locally/) for your system.

## To Do
[] : Fix loss zero near end of epoch
[] : Fix loss calculation

## Tmp

```
# Didn't work since prob_pos too similar always (so 1 or 0)
pos_goodness = torch.sum(torch.square(self.forward(x_pos)))  # [0;inf]
neg_goodness = torch.sum(torch.square(self.forward(x_neg)))  # [0;inf]

prob_pos_if_pos = torch.sigmoid(pos_goodness - self.threshold)  # [0.0067;1]
prob_pos_if_neg = torch.sigmoid(neg_goodness - self.threshold)  # [0.0067;1]

epsilon = 1e-7  # prevent log(0)

# Binary cross-entropy loss (ensure prob_pos_if_pos is close to 1 and prob_pos_if_neg is close to 0)
loss = -torch.log(prob_pos_if_pos + epsilon) - torch.log(
    1 - prob_pos_if_neg + epsilon
)  # [0;inf]
```

```
# Didn't work since positive will simply be slightly larger than negative (for layer2 and 3 only?)

goodness = torch.sum(torch.square(self.forward(x)))  # [0;inf]

if datatype == "pos":
    loss = -torch.log(goodness + epsilon)
elif datatype == "neg":
    loss = torch.log(goodness + epsilon)
```
