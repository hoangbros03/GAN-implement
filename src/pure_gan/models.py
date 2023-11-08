import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    """
    The Generator model class
    """

    def __init__(self, latent_dim, img_shape, activation="ReLU"):
        """Constructor

        Args:
            latent_dim (int): Dimension of latent space
            img_shape (tuple): Tuple containing shape of the image (channels, width, height)
            activation (str, optional): Define the activation. Defaults to "ReLU".
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        if activation=="ReLU":
            self.activation = nn.ReLU()
        elif activation=="LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Sigmoid()

        self.l1 = nn.Linear(latent_dim, 128)
        self.n0 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128,256)
        self.n1 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256,1024)
        self.n2 = nn.BatchNorm1d(1024)
        self.l4 = nn.Linear(1024,2048)
        self.n3 = nn.BatchNorm1d(2048)
        self.sigmoid = nn.Sigmoid()
        self.l_final = nn.Linear(2048, int(np.prod(img_shape)))
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, latent_space):
        """_summary_ TODO

        Args:
            latent_space (_type_): _description_
        """
        x = self.l1(latent_space)
        x = self.activation(x)
        x = self.n0(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.n1(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.n2(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.activation(x)
        x = self.n3(x)
        x = self.l_final(x)
        x = self.sigmoid(x)
        return x.reshape(-1, self.img_shape[1], self.img_shape[2])

class Discriminator(nn.Module):
    """
    The Discriminator class
    """

    def __init__(self, img_shape,activation="ReLU"):
        """_summary_ TODO

        Args:
            img_shape (_type_): _description_
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        if activation=="ReLU":
            self.activation = nn.ReLU()
        elif activation=="LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Sigmoid()
        
        self.l1 = nn.Linear(int(np.prod(img_shape)), 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.l2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.l4 = nn.Linear(64,1)
        self.drop_out = nn.Dropout(p=0.3)
        self.l_final = nn.Sigmoid()
    
    def forward(self, img):
        """_summary_ TODO

        Args:
            img (_type_): _description_
        """
        img = img.reshape(-1, self.img_shape[1] * self.img_shape[2])
        img = self.l1(img)
        img = self.activation(img)
        img = self.bn1(img)
        img = self.drop_out(img)
        img = self.l2(img)
        img = self.activation(img)
        img = self.bn2(img)
        img = self.l3(img)
        img = self.activation(img)
        img = self.bn3(img)
        img = self.drop_out(img)
        img = self.l4(img)
        img = self.l_final(img)
        return img