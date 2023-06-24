# The Forward-Forward Algorithm (Hinton, 2022)

**This code is for my personal educational purposes and still a work in progress. You are free to use it in any way you want, but know that there may be bugs.**

## Project description

This is a pytorch implementation of Hinton's [Forward-Forward algorithm](https://arxiv.org/abs/2212.13345)

The codebase is roughly based on mohammadpz's excellent [pytorch_forward_forward](https://github.com/mohammadpz/pytorch_forward_forward) implementation, with some major differences to keep it more in line with the sentiment of the original paper.
1. Model layers are not trained one by one, but all at once. One batch always passes through the entire network.
2. Positive and negative data is passed alternately through the network, instead of being mixed together.
3. The loss function is the one described in the paper.
4. Since I didn't see any implementation yet of the CNN model described in the paper, I implemented that version instead of the fully connected one.
5. I implemented both supervised and unsupervised versions.

The code in its current state behaves as described in the paper (i.e. high goodness for positive data and low goodness for negative data in each layer, while the overall loss and per layer loss decreases nicely). However, in the prediction phase, the **model does not perform as expected** yet and the accuracy on the test set is low. I'm still investigating why this is the case, feel free to open an issue if you have any ideas.

## Installation

```
python -m venv .venv
source .venv\bin\activate # or .venv\Scripts\activate.bat on Windows
pip install -r requirements.txt
```

Install latest version of [pytorch and torchvision](https://pytorch.org/get-started/locally/) for your system.

## Usage

You can train a network by running *main.py* with the following arguments:

```python
python main.py --supervised # train supervised model

python main.py --unsupervised_backbone # train unsupervised backbone

python main.py --unsupervised_clf # train unsupervised head (requires pretrained backbone). Will use latest model found in /models folder. Alternatively, provide the pretrained backbone filename with the argument --pretrained_backbone_filename
```
Check the *main.py* file for all available arguments.
Before training, the mnist datasets is downloaded while positive and negative data are generated and stored to disk (if it doesn't exist yet)
Trained models will be saved to the models/ folder.

All important information is logged to Tensorboard (losses, positive goodnesses, negative goodnesses) in the runs/ folder. 
For example, you can inspect the runs of a supervised model like this:
```bash
tensorboard --logdir runs/supervised # shows the supervised 
```

## Temporary notes

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
