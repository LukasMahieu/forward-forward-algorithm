"""Datasets and loaders for supervised learning example."""
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


def _create_positive_sample(image: torch.Tensor, label: int):
    """Create a positive sample by taking an image and replacing the first 10 pixels by a one of N representation of the label"""
    modified_image = image.clone()
    modified_image[:, 0, :10] = image.min()  # ensure no overlap with number
    modified_image[:, 0, label] = image.max()
    return modified_image


def _create_negative_sample(image: torch.Tensor, label: int):
    """Create a negative sample by taking an image and replacing the first 10 pixels by a one of N representation of a different label"""
    choices = list(range(10))
    choices.remove(label)
    dif_label = random.choice(choices)

    modified_image = image.clone()
    modified_image[:, 0, :10] = image.min()  # ensure no overlap with number
    modified_image[:, 0, dif_label] = image.max()
    return modified_image


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
    for i in tqdm(
        range(len(dataset)),
        desc="Creating positive dataset",
    ):
        image, label = dataset[i]
        positive_image = _create_positive_sample(image, label).squeeze()
        positive_image_path = os.path.join(
            positive_images_directory, f"positive_{i}.npy"
        )
        np.save(positive_image_path, positive_image.numpy())
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
        image, label = dataset[i]
        negative_image = _create_negative_sample(image, label).squeeze()

        negative_image_path = os.path.join(
            negative_images_directory, f"negative_{i}.npy"
        )
        np.save(negative_image_path, negative_image.numpy().astype(np.uint8))
    print("Negative data saved (+-230 MB).")


def create_mnist_datasets_unsupervised(root_path: str = "data"):
    """Load mnist and create positive and negative datasets."""
    train_mnist, test_mnist = _load_mnist(root_path)
    positive_path = os.path.join(root_path, "MNIST", "supervised", "positive_images")
    negative_path = os.path.join(root_path, "MNIST", "supervised", "negative_images")

    _save_positive_data(train_mnist, positive_path)
    _save_negative_data(train_mnist, negative_path)

    positive_train = SupervisedDataset(positive_path)
    negative_train = SupervisedDataset(negative_path)

    return positive_train, negative_train, test_mnist


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, images_directory: str):
        self.images_directory = images_directory
        self.images_list = os.listdir(images_directory)
        self.label = torch.Tensor([1 if "positive" in images_directory else 0])

    def __getitem__(self, index):
        image_path = os.path.join(self.images_directory, self.images_list[index])
        image = np.load(image_path)
        return image, self.label

    def __len__(self):
        return len(self.images_list)


if __name__ == "__main__":
    print("Testing supervised.py")
    train_dataset, test_dataset = _load_mnist("data")
    _save_positive_data(train_dataset, "data/MNIST/supervised/positive_images")
    _save_negative_data(train_dataset, "data/MNIST/supervised/negative_images")

    negative_dataset = SupervisedDataset("data/MNIST/supervised/negative_images")
    positive_dataset = SupervisedDataset("data/MNIST/supervised/positive_images")

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
