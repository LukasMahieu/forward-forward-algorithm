"""Datasets and loaders for unsupervised learning."""
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import create_mask, load_mnist


def save_positive_data(
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
            desc="Creating positive dataset. This will store around 230MB of data to local.",
        )
    ):
        positive_image_path = os.path.join(
            positive_images_directory, f"positive_{i}.npy"
        )
        np.save(positive_image_path, image.numpy())
    print("Positive data saved.")


def save_negative_data(
    dataset: torch.utils.data.Dataset, negative_images_directory: str
) -> None:
    """Create negative data following the method described in the paper and storing the images locally."""
    if os.path.exists(negative_images_directory):
        print("Negative data already exists, skipping...")
        return
    os.makedirs(negative_images_directory, exist_ok=True)
    for i in tqdm(
        range(len(dataset)),
        desc="Creating negative dataset. This will store around 230MB of data to local.",
    ):
        mask, mask_inv = create_mask(dataset.data[i].unsqueeze(0))

        # Pick a random image from the dataset that has a different target label
        indices = torch.nonzero(dataset.targets != dataset.targets[i]).squeeze()
        random_diff_digit = dataset.data[int(random.choice(indices))].unsqueeze(0)

        # Create new negative image
        negative_image = mask * dataset.data[i] + mask_inv * random_diff_digit[0]

        negative_image_path = os.path.join(
            negative_images_directory, f"negative_{i}.npy"
        )
        np.save(negative_image_path, negative_image.numpy())
    print("Negative data saved.")


class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, images_directory: str):
        self.images_directory = images_directory
        self.images_list = os.listdir(images_directory)
        self.label = 1 if "positive" in images_directory else 0

    def __getitem__(self, index):
        image_path = os.path.join(self.images_directory, self.images_list[index])
        image = np.load(image_path)
        return image, self.label

    def __len__(self):
        return len(self.images_list)


if __name__ == "__main__":
    print("Testing unsupervised.py")
    train_dataset, test_dataset = load_mnist()
    save_positive_data(train_dataset, "data/MNIST/positive_images")
    save_negative_data(train_dataset, "data/MNIST/negative_images")

    negative_dataset = UnsupervisedDataset("data/MNIST/negative_images")
    positive_dataset = UnsupervisedDataset("data/MNIST/positive_images")

    # Plot
    random_positive_index = torch.randint(0, len(positive_dataset), (1,))
    positive_image, pos_label = positive_dataset[random_positive_index]
    random_negative_index = torch.randint(0, len(negative_dataset), (1,))
    negative_image, neg_label = negative_dataset[random_negative_index]

    # Plot the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(positive_image.squeeze(), cmap="gray")
    ax[0].set_title(f"Positive")
    ax[0].axis("off")

    ax[1].imshow(negative_image.squeeze(), cmap="gray")
    ax[1].set_title("Negative")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
