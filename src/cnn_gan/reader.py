"""File to load Celeba data"""
import torch
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


class Reader:
    """Reader class"""

    def __init__(self, dataroot, batch_size=32, workers=2, image_size=64):
        """Constructor of the class

        Args:
            dataroot (str): path to the data directory
            batch_size (int, optional): Batch size number. Defaults to 32.
            workers (int, optional): Number of workers in CPU. Defaults to 2.
            image_size (int, optional): Size of the image. Defaults to 64.
        """

        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size

    def path_to_dataloader(self):
        """Get dataloader

        Returns:
            dataloader of dataset
        """
        dataset = dset.ImageFolder(
            root=self.dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )
        return dataloader

    def plot(self, device):
        """show images inside the dataloader

        Args:
            device (str): CUDA or CPU?
        """
        dataloader = self.path_to_dataloader()
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0].to(device)[:64], padding=2, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
        )
