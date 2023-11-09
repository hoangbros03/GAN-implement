""" 
File to get MNIST dataloader
"""
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import check_and_create_dir


def check_and_process_dataloader(directory, img_shape, batch_size):
    """Create MNIST dataloader

    Args:
        dir (string): Name of folder containing dataset
        img_shape: Shape of image
        batch_size (_type_): Batch size

    Returns:
        a dataloader of MNIST
    """
    check_and_create_dir(directory)
    transform = transforms.Compose(
        [
            transforms.Resize((img_shape[1], img_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    return DataLoader(
        datasets.MNIST(
            f"./{directory}/",
            train=True,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
