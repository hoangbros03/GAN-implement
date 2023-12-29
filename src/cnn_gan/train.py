"""Train file of enhanced GAN"""
import argparse

import torch
from torch import nn
from torch import optim
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from pure_gan.utils import check_and_create_dir
from pure_gan.mnist_dataloader import check_and_process_dataloader
from cnn_gan.layers import (
    Discriminator,
    Generator,
    Generator_Ablation,
    Discriminator_Ablation,
    GeneratorMNIST,
    DiscriminatorMNIST,
)
from cnn_gan.reader import Reader


class Trainer:
    """Trainer class"""

    def __init__(self, args):
        """Constructor of class Trainer

        Args:
            args (argparse): arguments specified when running file

        Raises:
            ValueError: generator must have type as defined
            ValueError: discriminator must have type as defined
        """
        # args.device, args.generator_type, args.discriminator_type, args.learning_rate
        self.args = args
        self.device = args.device
        generator = args.generator_type
        discriminator = args.discriminator_type
        lr = args.learning_rate
        channel = args.channel

        self.criterion = nn.BCELoss()
        self.channel = channel
        self.img_size = args.image_size
        args.image_size = int(args.image_size)
        if generator == "normal":
            self.netG = Generator(img_size=args.image_size, channel=channel).to(
                self.device
            )
        elif generator == "mnist":
            self.netG = GeneratorMNIST(img_size=args.image_size, channel=channel).to(
                self.device
            )
        elif generator == "ablation":
            self.netG = Generator_Ablation().to(self.device)
        else:
            raise ValueError("generator must be either normal or ablation")

        if discriminator == "normal":
            self.netD = Discriminator(img_size=args.image_size, channel=channel).to(
                self.device
            )
        elif discriminator == "mnist":
            self.netD = DiscriminatorMNIST(
                img_size=args.image_size, channel=channel
            ).to(self.device)
        elif discriminator == "ablation":
            self.netD = Discriminator_Ablation().to(self.device)
        else:
            raise ValueError("discriminator must be either normal or ablation")

        # Establish convention for real and fake labels during training
        self.real_label = 1.0
        self.fake_label = 0.0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        self.pred = []
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

    def train_loop(self, dataloader):
        """Trainning loop

        Args:
            dataloader (_type_): The dataloader
        """
        # Set up variables
        num_epochs = self.args.epochs
        noise_size = self.args.noise_size
        b_size = self.args.batch_size
        D_G_train_proportion = self.args.proportion

        fixed_noise = torch.randn(b_size, noise_size, 1, 1, device=self.device)

        # Lists to keep track of progress
        iters = 0

        # print("Starting Training Loop...")
        # For each epochs

        for epoch in tqdm(range(num_epochs)):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                if i % D_G_train_proportion == 0:
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    # print("\n=============DISCRIMINATOR TRAINING=============\n")
                    # print("Training true batch...")
                    self.netD.zero_grad()
                    # Format batch
                    real_cpu = data[0].to(self.device)
                    b_size = real_cpu.size(0)
                    label = torch.full(
                        (b_size,),
                        self.real_label,
                        dtype=torch.float,
                        device=self.device,
                    )
                    real_cpu.view(b_size, self.img_size, self.img_size, self.channel)
                    # print("Forwarding batch...")
                    # Forward pass real batch through D
                    output = self.netD(real_cpu).squeeze()
                    # Calculate loss on all-real batch
                    # print("Calculating loss...")
                    # print(f"Shape: Output: {output.shape}, label: {label.shape}")
                    errD_real = self.criterion(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    # D_x = output.mean().item()

                    # print("\nTraining fake batch...")
                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, noise_size, 1, 1, device=self.device)
                    # Generate fake image batch with G
                    # print("Generating...")
                    fake = self.netG(noise)
                    # print(f"Fake shape: {fake.shape}")
                    label.fill_(self.fake_label)
                    # Classify all fake batch with D
                    # print("Classfying with D...")
                    output = self.netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    # print("Calculating loss...")
                    # print(f"Shape: Output: {output.shape}, label: {label.shape}")
                    errD_fake = self.criterion(output, label)
                    # Calculate the gradients for this batch,
                    # accumulated (summed) with previous gradients
                    # print("Backward pass...")
                    errD_fake.backward()
                    # D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    self.optimizerD.step()
                    # print("Done training discriminator...")

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                # print("\n=============GENERATOR TRAINING=============\n")
                noise = torch.randn(b_size, noise_size, 1, 1, device=self.device)
                # Generate fake image batch with G
                # print("Generating...")
                fake = self.netG(noise)
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                # print("Generating...")
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                # print("Calculating G's loss...")
                errG = self.criterion(output, label)
                # Calculate gradients for G
                # print("Backward pass...")
                errG.backward()
                # D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()
                # print("Done training generator...")

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                if bool(self.args.key):
                    wandb.log(
                        {
                            "g_loss": errG.item(),
                            "d_loss": errD.item(),
                        }
                    )

                # print("Generating on fixed noise...")
                # Check how the generator is doing by saving G's output on fixed_noise
                # Reduce time append to save the ram usage
                if (iters % 10 == 0) and (
                    (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )
                    self.pred.append(
                        self.netG(
                            torch.randn(128, noise_size, 1, 1, device=self.device)
                        )
                        .detach()
                        .cpu()
                    )

                iters += 1

            if self.args.save_model:
                if (
                    epoch + 1
                ) % self.args.save_frequency == 0 or epoch == num_epochs - 1:
                    check_and_create_dir(self.args.output_dir)
                    torch.save(
                        self.netG.state_dict(),
                        f"{self.args.output_dir}/{str(self.args.generator_type)}\
                            _{str(self.args.discriminator_type)}_{str(epoch+1)}.pth",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument(
        "-i", "--image_size", help="size of the images", type=int, default=64
    )
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("-n", "--noise_size", help="noise size", type=int, default=100)
    parser.add_argument(
        "-lr", "--learning_rate", help="learning rate", type=float, default=0.0001
    )
    parser.add_argument(
        "-p", "--proportion", help="proportion between D and G", type=int, default=3
    )
    parser.add_argument(
        "-dt",
        "--discriminator_type",
        help="type of discriminator",
        type=str,
        default="normal",
    )
    parser.add_argument(
        "-gt", "--generator_type", help="type of generator", type=str, default="normal"
    )
    parser.add_argument("-d", "--device", help="cpu or cuda?", type=str, default="cpu")
    parser.add_argument("-dr", "--data_root", help="data root", type=str, default="data")
    parser.add_argument("-k", "--key", help="key of wandb", type=str, default=None)
    parser.add_argument(
        "-o",
        "--output_dir",
        help="directory of the output",
        type=str,
        default="output_models",
    )
    parser.add_argument(
        "-s", "--save_model", help="save model or not?", action="store_true"
    )
    parser.add_argument(
        "-nk", "--num_workers", help="number of worker cpu", type=int, default=2
    )
    parser.add_argument(
        "-sf", "--save_frequency", help="frequency of saving model", type=int, default=3
    )
    parser.add_argument(
        "-c", "--channel", help="number of channel in the image", type=int, default=3
    )
    input_args = parser.parse_args()

    input_log = False
    if input_args.key is not None and input_args.key != "None":
        # Wandb
        wandb.login(key=input_args.key)
        run = wandb.init(
            # Set the project where this run will be logged
            project="cnn-gan",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": input_args.learning_rate,
                "epochs": input_args.epochs,
                "discriminator_type": input_args.discriminator_type,
                "generator_type": input_args.generator_type,
            },
        )
        input_log = True

    trainer = Trainer(input_args)
    if input_args.discriminator_type != "mnist":
        the_dataloader = Reader(
            input_args.data_root,
            input_args.batch_size,
            input_args.num_workers,
            input_args.image_size,
        ).path_to_dataloader()
    else:
        the_dataloader = check_and_process_dataloader("mnist", (1, 28, 28), 32)
    trainer.train_loop(the_dataloader)
