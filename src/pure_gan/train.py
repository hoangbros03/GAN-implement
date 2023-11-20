""" 
Train file
"""
import argparse
import datetime

import torch
from torch import nn
from torch.optim import Adam
from pure_gan.models import Generator, Discriminator
from pure_gan.utils import check_and_create_dir, get_random_string
from pure_gan.mnist_dataloader import check_and_process_dataloader
import wandb


def train(dataloader, epochs, latent_dim, img_shape, learning_rate, output_model_dir):
    """Train function

    Args:
        dataloader (Dataloader): The dataloader
        epochs (int): Number of epochs
        latent_dim (int): Dim of latent space
        img_shape (tuple): Shape of image
        learning_rate (float): The learning rate
        output_model_dir (string): Name of folder containing models

    """
    # Random string
    code_random = str(get_random_string(7))

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss and model

    generator_model = Generator(latent_dim, img_shape, activation="LeakyReLU").to(
        device
    )
    disciminator_model = Discriminator(img_shape, activation="LeakyReLU").to(device)

    g_optimizer = Adam(generator_model.parameters(), lr=learning_rate)
    d_optimizer = Adam(disciminator_model.parameters(), lr=learning_rate)

    value_function_loss = nn.BCELoss()

    def train_discriminator():
        """Main function to train the discriminator

        Returns:
            loss of discriminator
        """
        # Get ground truth
        real_ground_truth = 0.3 * torch.rand(imgs.shape[0]) + 0.7
        real_ground_truth = real_ground_truth.to(device)
        fake_ground_truth = 0.3 * torch.rand(imgs.shape[0])
        fake_ground_truth = fake_ground_truth.to(device)

        # Train discriminator
        d_optimizer.zero_grad()

        output_real = disciminator_model(imgs).view(-1)
        real_loss = value_function_loss(output_real, real_ground_truth)

        latent_space = torch.randn(imgs.shape[0], latent_dim).to(device)
        fake_samples = generator_model(latent_space).detach()
        output_fake = disciminator_model(fake_samples).view(-1)
        # print(f"Fake sample shape: {fake_samples.shape}")
        fake_loss = value_function_loss(output_fake, fake_ground_truth)
        real_loss.backward()
        fake_loss.backward()
        d_optimizer.step()
        return real_loss + fake_loss

    def train_generator():
        """Main function to train the generator

        Returns:
            loss of generator
        """
        g_optimizer.zero_grad()
        real_ground_truth = 0.3 * torch.rand(imgs.shape[0]) + 0.7
        real_ground_truth = real_ground_truth.to(device)
        latent_space = torch.randn(imgs.shape[0], latent_dim) * 1.0
        fake_samples = generator_model(latent_space.to(device))
        g_loss = value_function_loss(
            disciminator_model(fake_samples).view(-1), real_ground_truth
        )
        g_loss.backward()
        g_optimizer.step()

        return g_loss

    for epoch in range(epochs):
        imgs = None
        total_loss_d = 0.0
        total_loss_g = 0.0
        total_ite = 0
        for ite, (imgs, _) in enumerate(dataloader):
            generator_model.train()
            disciminator_model.train()
            imgs = imgs.reshape((-1, 1, img_shape[1], img_shape[2])).to(device)
            # print(imgs.shape)
            for _ in range(1):
                # Train the discriminator
                total_loss_d += train_discriminator()

            # Train the generator
            total_loss_g += train_generator()
            total_ite = ite
        total_ite += 1
        print(
            f"Epoch: {epoch}/{epochs}, g_loss: {total_loss_g/total_ite}, \
                d_loss: {total_loss_d/total_ite}"
        )
        wandb.log(
            {
                "Epoch": epoch,
                "Total epoch": epochs,
                "g_loss": total_loss_g / total_ite,
                "d_loss": total_loss_d / total_ite,
            }
        )

        # Output the model
        if (epoch + 1) % 10 == 0:
            torch.save(
                generator_model.state_dict(),
                f"{output_model_dir}/{str(code_random)}_{str(epoch+1)}.pth",
            )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--epochs", type=int, default=200, help="number of epochs"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--output_model_dir", type=str, default="models", help="Output model dir"
    )
    args = parser.parse_args()

    # CONSTANT VARIABLE
    IMG_SHAPE = (1, 28, 28)

    # Create dir
    datetime_now = datetime.datetime.now().isoformat()
    check_and_create_dir(args.output_model_dir)

    # Dataloader
    mnist_dataloader = check_and_process_dataloader("mnist", IMG_SHAPE, args.batch_size)

    # Wandb
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="new-pure-gan",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
        },
    )

    # Train
    train(
        mnist_dataloader,
        args.epochs,
        args.latent_dim,
        IMG_SHAPE,
        args.lr,
        args.output_model_dir,
    )
