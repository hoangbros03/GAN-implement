import argparse
from pathlib import Path
import sys
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models import Generator, Discriminator
from utils import check_and_create_dir, get_random_string
from mnist_dataloader import check_and_process_dataloader
import wandb

def train(dataloader, epochs, latent_dim, img_shape, batch_size, learning_rate, output_model_dir):
    """_summary_ TODO

    Args:
        dataloader (_type_): _description_
    """
    # Random string
    code_random = str(get_random_string(7))

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loss and model
    value_function_loss = nn.BCELoss()
    generator_model = Generator(latent_dim,img_shape, activation="LeakyReLU").to(device)
    disciminator_model = Discriminator(img_shape, activation="LeakyReLU").to(device)

    g_optimizer = Adam(generator_model.parameters(), lr=learning_rate)
    d_optimizer = Adam(disciminator_model.parameters(), lr=learning_rate)
    
    generator_model.train()
    disciminator_model.train()
    for epoch in range(epochs):
        for _, (imgs, _) in enumerate(dataloader):
            imgs = imgs.reshape((-1, img_shape[1], img_shape[2])).to(device)

            # Train the generator
            g_optimizer.zero_grad()
            real_ground_truth = 0.3 * torch.rand(imgs.shape[0],1) + 0.7
            real_ground_truth.to(device)
            latent_space = torch.randn(imgs.shape[0],100) * 1.0
            fake_samples = generator_model(latent_space.to(device))
            g_loss = value_function_loss(disciminator_model(fake_samples), real_ground_truth)
            g_loss.backward()
            g_optimizer.step()
        # Get ground truth
        real_ground_truth = 0.3 * torch.rand(imgs.shape[0],1) + 0.7
        real_ground_truth.to(device)
        fake_ground_truth = 0.3 * torch.rand(imgs.shape[0],1)
        fake_ground_truth.to(device)

        # Train discriminator
        d_optimizer.zero_grad()
        latent_space = torch.randn(imgs.shape[0],100) * 1.0
        fake_samples = generator_model(latent_space.to(device))

        real_loss = value_function_loss(disciminator_model(imgs), real_ground_truth)
        fake_loss = value_function_loss(disciminator_model(fake_samples), fake_ground_truth)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        d_optimizer.step()


        print(f"Epoch: {epoch}/{epochs}, g_loss: {g_loss.item()}, d_loss: {d_loss.item()}")
        wandb.log({
          "Epoch": epoch, "Total epoch": epochs, "g_loss": g_loss.item(), "d_loss": d_loss.item()
            })
       
        # Output the model
        if (epoch+1) % 50 == 0:
            torch.save(generator_model.state_dict(), f"{output_model_dir}/{str(code_random)}_{str(epoch+1)}.pth")

        
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-b","--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--cpus", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--output_model_dir", type=str, default="models", help="Output model dir")
    args = parser.parse_args()

    # CONSTANT VARIABLE
    IMG_SHAPE = (1,28,28)
    IMG_SHAPE = (1,28,28)


    # Create dir
    datetime_now = datetime.datetime.now().isoformat()
    check_and_create_dir(args.output_model_dir)

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
        "learning_rate": args.lr,
        "epochs": args.epochs,
    })

    # Train
    train(dataloader, args.epochs, args.latent_dim, IMG_SHAPE, args.batch_size, args.lr, args.output_model_dir)
    
