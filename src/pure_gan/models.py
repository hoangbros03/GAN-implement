""" 
File contains the models
"""
import numpy as np
from torch import nn


class Generator(nn.Module):
    """
    The Generator model class
    """

    def __init__(self, latent_dim, img_shape, activation="ReLU"):
        """Constructor

        Args:
            latent_dim (int): Dimension of latent space
            img_shape (tuple):
                Tuple containing shape of the image (channels, width, height)
            activation (str, optional): Define the activation. Defaults to "ReLU".
        """
        super().__init__()
        self.img_shape = img_shape
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Sigmoid()

        self.l1 = nn.Linear(latent_dim, 128)
        self.n0 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 256)
        self.n1 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256, 1024)
        self.n2 = nn.BatchNorm1d(1024)
        self.l4 = nn.Linear(1024, 2048)
        self.n3 = nn.BatchNorm1d(2048)
        self.tanh = nn.Tanh()
        self.l_final = nn.Linear(2048, int(np.prod(img_shape)))
        self.dropout = nn.Dropout(p=0.3)
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, latent_space):
        """Forward function

        Args:
            latent_space (torch): The latent space
        """
        
        return self.main(latent_space).view(-1, 1, self.img_shape[1], self.img_shape[2])


class Discriminator(nn.Module):
    """
    The Discriminator class
    """

    def __init__(self, img_shape, activation="ReLU"):
        """_summary_ TODO

        Args:
            img_shape (): shape of image
            activation (string): Name of activation

        Returns:
            The output of model
        """
        super().__init__()
        self.img_shape = img_shape

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Sigmoid()

        self.main = nn.Sequential(
            nn.Linear(self.img_shape[1] * self.img_shape[2], 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        """Forward function

        Args:
            img (_type_): The input

        Returns:
            The output of model
        """
        img = img.view(-1, self.img_shape[1] * self.img_shape[2])
        return self.main(img)
