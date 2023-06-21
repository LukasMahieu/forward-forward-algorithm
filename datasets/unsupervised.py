"""Datasets and loaders for unsupervised learning example."""
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils.data import DataLoader

################### Helper Functions ###################


def _create_mask(
    image_tensor: torch.Tensor, n_repeats: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a mask and its inverse for the image.

    Start with random bit image and repeatedly blur with a filter of form [1/4, 1/2, 1/4] in both horizontal and vertical direction. Threshold the image to get a binary mask.

    Note: I played around with this and found that 8 repeats gives a mask that makes the most sense. I do wonder if it's not better to start with a positive masks that ensures larger regions of 1's than 0's, which is not always the case here.

    Args:
        image_tensor (torch.Tensor): image tensor (1x28x28)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: mask tensor, inverse of mask tensor
    """
    # Create random bit image
    mask = torch.randint(0, 2, image_tensor.shape).float()

    # Blur the image
    for _ in range(n_repeats):
        mask = F.conv2d(
            mask,
            torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3, 1),
            padding=(0, 1),
        )
        mask = F.conv2d(
            mask,
            torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 1, 3),
            padding=(1, 0),
        )

    mask = (mask > 0.5).float()  # threshold
    mask_inv = 1 - mask  # inverse

    return mask, mask_inv


def _load_mnist(root: str = "data"):
    """Load MNIST dataset.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: train dataset, test dataset
    """
    print("Loading MNIST dataset...")
    # Transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load datasets
    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


################### Main dataset functions/classes ###################


def _save_positive_data(
    dataset: torch.utils.data.Dataset, positive_images_directory: str
) -> None:
    """Save positive data to local."""
    if os.path.exists(positive_images_directory):
        print("Positive data already exists, skipping...")
        return
    os.makedirs(positive_images_directory, exist_ok=True)
    for i, image in enumerate(
        tqdm(
            dataset.data,
            desc="Creating positive dataset",
        )
    ):
        positive_image_path = os.path.join(
            positive_images_directory, f"positive_{i}.npy"
        )
        np.save(positive_image_path, image.numpy())
    print("Positive data saved (+-230 MB).")


def _save_negative_data(
    dataset: torch.utils.data.Dataset, negative_images_directory: str
) -> None:
    """Create negative data following the method described in the paper and storing the images locally."""
    if os.path.exists(negative_images_directory):
        print("Negative data already exists, skipping...")
        return
    os.makedirs(negative_images_directory, exist_ok=True)
    for i in tqdm(
        range(len(dataset)),
        desc="Creating negative dataset",
    ):
        mask, mask_inv = _create_mask(dataset.data[i].unsqueeze(0))

        # Pick a random image from the dataset that has a different target label
        indices = torch.nonzero(dataset.targets != dataset.targets[i]).squeeze()
        random_diff_digit = dataset.data[int(random.choice(indices))].unsqueeze(0)

        # Create new negative image
        negative_image = mask * dataset.data[i] + mask_inv * random_diff_digit[0]
        negative_image = negative_image.squeeze()  # remove channel dimension

        negative_image_path = os.path.join(
            negative_images_directory, f"negative_{i}.npy"
        )
        np.save(negative_image_path, negative_image.numpy().astype(np.uint8))
    print("Negative data saved (+-230 MB).")


def create_mnist_datasets(root_path: str = "data"):
    """Load mnist and create positive and negative datasets."""
    train_mnist, test_mnist = _load_mnist(root_path)
    positive_path = os.path.join(root_path, "MNIST", "positive_images")
    negative_path = os.path.join(root_path, "MNIST", "negative_images")

    _save_positive_data(train_mnist, positive_path)
    _save_negative_data(train_mnist, negative_path)

    positive_train = UnsupervisedDataset(positive_path)
    negative_train = UnsupervisedDataset(negative_path)

    return positive_train, negative_train, test_mnist


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, images_directory: str):
        self.images_directory = images_directory
        self.images_list = os.listdir(images_directory)
        self.label = torch.Tensor([1 if "positive" in images_directory else 0])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __getitem__(self, index):
        image_path = os.path.join(self.images_directory, self.images_list[index])
        image = np.load(image_path)
        image = self.transform(image)
        return image, self.label

    def __len__(self):
        return len(self.images_list)


if __name__ == "__main__":
    print("Testing unsupervised.py")
    train_dataset, test_dataset = _load_mnist("data")
    _save_positive_data(train_dataset, "data/MNIST/positive_images")
    _save_negative_data(train_dataset, "data/MNIST/negative_images")

    negative_dataset = UnsupervisedDataset("data/MNIST/negative_images")
    positive_dataset = UnsupervisedDataset("data/MNIST/positive_images")

    # Create data loaders for the datasets
    negative_loader = DataLoader(negative_dataset, batch_size=1, shuffle=True)
    positive_loader = DataLoader(positive_dataset, batch_size=1, shuffle=True)

    # Get 5 random negative images
    negative_images = []
    for _ in range(5):
        random_negative_index = torch.randint(0, len(negative_loader.dataset), (1,))
        negative_image, neg_label = negative_loader.dataset[random_negative_index]
        negative_images.append(negative_image)

    # Get 5 random positive images
    positive_images = []
    for _ in range(5):
        random_positive_index = torch.randint(0, len(positive_loader.dataset), (1,))
        positive_image, pos_label = positive_loader.dataset[random_positive_index]
        positive_images.append(positive_image)

    # Plot
    fig, ax = plt.subplots(5, 2, figsize=(10, 20))

    for i in range(5):
        ax[i, 0].imshow(negative_images[i].squeeze(), cmap="gray")
        ax[i, 0].set_title(f"Negative {i+1}")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(positive_images[i].squeeze(), cmap="gray")
        ax[i, 1].set_title(f"Positive {i+1}")
        ax[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
