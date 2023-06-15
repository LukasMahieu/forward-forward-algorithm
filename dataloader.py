"""Load data from different datasets and helper functions for displaying data."""

import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

########## Helper functions ##########


def _create_mask(
    image_tensor: torch.Tensor, n_repeats: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a mask and its inverse for the image.

    Start with random bit image and repeatedly blur with a filter of form [1/4, 1/2, 1/4]
    in both horizontal and vertical direction. Threshold the image to get a binary mask.

    Note: I played around with this and found that 8 repeats gives a mask that makes the
    most sense. I do wonder if it's not better to start with a positive masks that ensures
    larger regions of 1's than 0's, which is not always the case here.

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

    # Threshold the mask
    mask = (mask > 0.5).float()

    # Inverse of mask
    mask_inv = 1 - mask

    return mask, mask_inv


########## Datasets ##########


def create_datasets_mnist(
    type: str = "unsupervised",
    data_path: str = "data",
) -> tuple[DataLoader, DataLoader]:
    """Load train and test data from MNIST.

    Args:
        type (str): type of dataset, e.g. 'unsupervised' or 'supervised'
        data_path (str): root folder to save data, e.g. 'data'
        batch_size (int): batch size
        num_workers (int): number of workers

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: pytorch DataLoaders for
        positive (train), negative (train) and test set
    """
    train_test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    positive_dataset = MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=train_test_transforms,
    )

    negative_dataset = MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=train_test_transforms,
    )

    test_dataset = MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=train_test_transforms,
    )

    if type == "unsupervised":
        # Create positive dataset
        positive_dataset.targets = torch.ones(
            len(positive_dataset.targets)
        )  # Change the labels to 1 (positive)

        # Create negative dataset
        original_images = negative_dataset.data.clone()
        original_labels = negative_dataset.targets.clone()

        for i in tqdm(
            range(len(negative_dataset)), desc="Creating negative dataset"
        ):

            # Create mask
            mask, mask_inv = _create_mask(original_images[i].unsqueeze(0))

            # Pick a random image from the dataset that has a different target label
            indices = original_labels != original_labels[i]
            indices = torch.nonzero(indices).squeeze()
            random_index = int(random.choice(indices))
            random_diff_digit = original_images[random_index].unsqueeze(0)

            # Create new negative image
            negative_dataset.data[i] = (
                mask * original_images[i] + mask_inv * random_diff_digit[0]
            )

        negative_dataset.targets = torch.zeros(
            len(negative_dataset.targets)
        )  # Change the labels to 0 (negative)

    elif type == "supervised":
        # Create positive dataset
        pass

    return positive_dataset, negative_dataset, test_dataset


def load_data(
    positive_data,
    negative_data,
    test_data,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Create torch dataloaders for positive, negative and test set.

    Args:
        positive_data (torch.utils.data.Dataset): positive dataset
        negative_data (torch.utils.data.Dataset): negative dataset
        test_data (torch.utils.data.Dataset): test dataset
        batch_size (int): batch size. Defaults to 8.
        num_workers (int): Number of workers. Defaults to 0.
    """

    positive_loader = DataLoader(
        positive_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    negative_loader = DataLoader(
        negative_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return positive_loader, negative_loader, test_loader


# Test the dataloader
if __name__ == "__main__":
    print("Testing the dataloader...")
    positive_dataset, negative_dataset, test_dataset = create_datasets_mnist(
        type="unsupervised"
    )

    positive_loader, negative_loader, test_loader = load_data(
        positive_dataset, negative_dataset, test_dataset
    )

    fig, ax = plt.subplots(1, 2)

    for batch_idx, (data, target) in enumerate(positive_loader):
        print(data.shape, target.shape)
        ax[0].imshow(data[0][0], cmap="gray")
        break

    for batch_idx, (data, target) in enumerate(negative_loader):
        print(data.shape, target.shape)
        ax[1].imshow(data[0][0], cmap="gray")
        break

    plt.show()

    print("Done!")
