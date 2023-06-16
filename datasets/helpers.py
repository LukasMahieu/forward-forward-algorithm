import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST


def create_mask(
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


def load_mnist(root: str = "data"):
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
