import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import check_and_create_dir
from mnist_dataloader import check_and_process_dataloader
import wandb

class Generator(nn.Module):
    """
    The Generator model class
    """

    def __init__(self, latent_dim, img_shape, batch_size, activation="ReLU"):
        """Constructor

        Args:
            latent_dim (int): Dimension of latent space
            img_shape (tuple): Tuple containing shape of the image (channels, width, height)
            activation (str, optional): Define the activation. Defaults to "ReLU".
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.batch_size = batch_size
        if activation=="ReLU":
            self.activation = nn.ReLU()
        elif activation=="LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Sigmoid()

        self.l1 = nn.Linear(latent_dim, 128)
        self.l2 = nn.Linear(128,256)
        self.n1 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256,1024)
        self.n2 = nn.BatchNorm1d(1024)
        self.l4 = nn.Linear(1024,2048)
        self.n3 = nn.BatchNorm1d(2048)
        self.l_final = nn.Linear(2048, int(np.prod(img_shape)))
    
    def forward(self, latent_space):
        """_summary_ TODO

        Args:
            latent_space (_type_): _description_
        """
        x = self.l1(latent_space)
        x = self.l2(x)
        x = self.n1(x)
        x = self.l3(x)
        x = self.n2(x)
        x = self.l4(x)
        x = self.n3(x)
        x = self.l_final(x)
        x = self.activation(x)
        return x.reshape(-1, self.img_shape[1], self.img_shape[2])

class Discriminator(nn.Module):
    """
    The Discriminator class
    """

    def __init__(self, img_shape, batch_size,activation="ReLU"):
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
        self.l2 = nn.Linear(512,256)
        self.l3 = nn.Linear(256,64)
        self.l4 = nn.Linear(64,1)
        self.l_final = nn.Sigmoid()
    
    def forward(self, img):
        """_summary_ TODO

        Args:
            img (_type_): _description_
        """
        img = img.reshape(-1, self.img_shape[1] * self.img_shape[2])
        img = self.l1(img)
        img = self.l2(img)
        img = self.l3(img)
        img = self.l4(img)
        img = self.l_final(img)
        return img

def train(dataloader, epochs, latent_dim, img_shape, batch_size, learning_rate):
    """_summary_ TODO

    Args:
        dataloader (_type_): _description_
    """
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loss and model
    value_function_loss = nn.BCELoss()
    generator_model = Generator(latent_dim,img_shape,batch_size)
    disciminator_model = Discriminator(img_shape,batch_size)

    g_optimizer = Adam(generator_model.parameters(), lr=learning_rate)
    d_optimizer = Adam(disciminator_model.parameters(), lr=learning_rate)
    
    generator_model.to(device)
    disciminator_model.to(device)
    generator_model.train()
    disciminator_model.train()
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.reshape((-1, img_shape[1], img_shape[2]))
            # Get ground truth
            real_ground_truth = torch.ones(imgs.shape[0], 1)
            fake_ground_truth = torch.zeros(imgs.shape[0], 1)

            # Train discriminator
            imgs.to(device)
            d_optimizer.zero_grad()
            fake_samples = generator_model(torch.randint(0,2,(imgs.shape[0],latent_dim)).float())

            real_loss = value_function_loss(disciminator_model(imgs), real_ground_truth)
            fake_loss = value_function_loss(disciminator_model(fake_samples), fake_ground_truth)
            d_loss = (real_loss + fake_loss)/2
            print(f"epoch: {epoch}/{epochs}, d_loss: {d_loss.item()}")
            wandb.log({
                "Epoch": epoch,
                "D_loss": d_loss.item()
            })
            d_loss.backward()
            d_optimizer.step()


        # Train the generator
        g_optimizer.zero_grad()
        fake_samples = generator_model(torch.randint(0,2,(imgs.shape[0],latent_dim)).float())
        g_loss = value_function_loss(disciminator_model(fake_samples), real_ground_truth)
        g_loss.backward()
        g_optimizer.step()

        print(f"Epoch: {epoch}/{epochs}, g_loss: {g_loss.item()}")
        wandb.log({
          "Epoch": epoch, "Total epoch": epochs, "g_loss": g_loss.item()
            })

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--cpus", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    args = parser.parse_args()

    # CONSTANT VARIABLE
    IMG_SHAPE = (1,28,28)

    # Create dir
    check_and_create_dir("gen_images")

    # Dataloader
    dataloader = check_and_process_dataloader("mnist", IMG_SHAPE, args.batch_size)

    # Wandb
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
    project="pure-gan",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    })

    # Train
    train(dataloader, args.epochs, args.latent_dim, IMG_SHAPE, args.batch_size, args.lr)
    
