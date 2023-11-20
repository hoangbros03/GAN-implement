"""File contains models"""
from torch import nn


class Generator(nn.Module):
    """Generator class"""

    def __init__(self, noise_size=100, img_size=64, channel=3):
        """Constructor of the class

        Args:
            noise_size (int, optional): As described in name. Defaults to 100.
            img_size (int, optional): As described in name. Defaults to 64.
            channel (int, optional): As described in name. Defaults to 3.
        """

        super().__init__()
        self.ns = noise_size
        self.imgs = img_size
        self.c = channel
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ns, self.imgs * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.imgs * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 8, self.imgs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.imgs * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 4, self.imgs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.imgs * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 2, self.imgs * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.imgs * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs, self.c, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output


class Discriminator(nn.Module):
    """Discriminator class"""

    def __init__(self, img_size=64, channel=3):
        """Constructor of this class

        Args:
            img_size (int, optional): As described in name. Defaults to 64.
            channel (int, optional): As described in name. Defaults to 3.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output


class GeneratorMNIST(nn.Module):
    """Generator MNIST class"""

    def __init__(self, noise_size=100, img_size=64, channel=3):
        """Constructor of the class

        Args:
            noise_size (int, optional): As described in name. Defaults to 100.
            img_size (int, optional): As described in name. Defaults to 64.
            channel (int, optional): As described in name. Defaults to 3.
        """

        super().__init__()
        self.ns = noise_size
        self.imgs = img_size
        self.c = channel
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ns, self.imgs * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.imgs * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 4, self.imgs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.imgs * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 2, self.imgs * 1, 4, 2, 2, bias=False),
            nn.BatchNorm2d(self.imgs * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 1, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output


class DiscriminatorMNIST(nn.Module):
    """Discriminator MNIST class"""

    def __init__(self, img_size=64, channel=3):
        """Constructor of the class

        Args:
            img_size (int, optional): As described in name. Defaults to 64.
            channel (int, optional): As described in name. Defaults to 3.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel, img_size, 4, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size, img_size * 2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output


class Generator_Ablation(nn.Module):
    """Generator ablation class"""

    def __init__(self, noise_size=100, img_size=64, channel=3):
        """Constructor of the class

        Args:
            noise_size (int, optional): As described in name. Defaults to 100.
            img_size (int, optional): As described in name. Defaults to 64.
            channel (int, optional): As described in name. Defaults to 3.
        """

        super().__init__()
        self.ns = noise_size
        self.imgs = img_size
        self.c = channel
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ns, self.imgs * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 8, self.imgs * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 4, self.imgs * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs * 2, self.imgs * 1, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.imgs, self.c, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output


class Discriminator_Ablation(nn.Module):
    """Discriminator ablation class"""

    def __init__(self, img_size=64):
        """Constructor of the class

        Args:
            img_size (int, optional): As described in name. Defaults to 64.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        """Forward function

        Args:
            input_tensor (torch.Tensor): input_tensor tensor

        Returns:
            A output tensor
        """
        output = self.main(input_tensor)
        return output
