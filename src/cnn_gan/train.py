import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import wandb
from layers import Discriminator, Generator, Generator_Ablation, Discriminator_Ablation
from reader import Reader

class Trainer:
    def __init__(self, device='cpu', generator="normal", discriminator="normal", lr=0.0001, log=False):
        self.device = device
        self.criterion = nn.BCELoss()
        if generator == "normal":
            self.netG = Generator().to(device)
        elif generator == "ablation":
            self.netG = Generator_Ablation().to(device)
        else:
            raise ValueError("generator must be either normal or ablation")

        if discriminator == "normal":
            self.netD = Discriminator().to(device)
        elif discriminator == "ablation":
            self.netD = Discriminator_Ablation().to(device)
        else:
            raise ValueError("discriminator must be either normal or ablation")

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        self.img_list = []
        self.G_losses = []
        self.D_losses = []

    def train_loop(self,
                   dataloader,
                   num_epochs=1,
                   img_size=64,
                   noise_size=100,
                   b_size=32,
                   channel=3,
                   D_G_train_proportion=3):
        
        fixed_noise = torch.randn(b_size, noise_size, 1, 1, device=self.device)

        # Lists to keep track of progress
        iters = 0

        # print("Starting Training Loop...")
        # For each epoch
        from tqdm import tqdm
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
                    label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                    real_cpu.view(b_size, img_size, img_size, channel)
                    # print("Forwarding batch...")
                    # Forward pass real batch through D
                    output = self.netD(real_cpu).squeeze()
                    # Calculate loss on all-real batch
                    # print("Calculating loss...")
                    errD_real = self.criterion(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()

                    # print("\nTraining fake batch...")
                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, noise_size, 1, 1, device=self.device)
                    # Generate fake image batch with G
                    # print("Generating...")
                    fake = self.netG(noise)
                    label.fill_(self.fake_label)
                    # Classify all fake batch with D
                    # print("Classfying with D...")
                    output = self.netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    # print("Calculating loss...")
                    errD_fake = self.criterion(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    # print("Backward pass...")
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
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
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()
                # print("Done training generator...")

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                
                if log:
                    print(f"Epoch: {epoch}, total epoch: {num_epochs} \
                    g_loss: {errG.item()}, d_loss: {errD.item()}")
                    wandb.log(
                    {
                        "Epoch": epoch,
                        "Total epoch": num_epochs,
                        "g_loss": errG.item(),
                        "d_loss": errD.item(),
                    }
                    )

                # print("Generating on fixed noise...")
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("-i", "--image_size", help="size of the images", type=int, default=64)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("-n", "--noise_size", help="noise size", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.0001)
    parser.add_argument("-p", "--proportion", help="proportion between D and G", type=int, default=3)
    parser.add_argument("-dt", "--discriminator_type", help="type of discriminator", type=str, default="normal")
    parser.add_argument("-gt", "--generator_type", help="type of generator", type=str, default="normal")
    parser.add_argument("d", "--device", help="cpu or cuda?", type=str, default="cpu")
    parser.add_argument("-dr", "--data_root", help="data root", type=str, required=True)
    parser.add_argument("-k","--key",help="key of wandb", type=str, default=None)

    log = False
    if args.key is not None:
        # Wandb
        wandb.login(key=args.key)
        run = wandb.init(
            # Set the project where this run will be logged
            project="unet-gan",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "discriminator_type": args.discriminator_type,
                "generator_type": args.generator_type,
            },
        )
        log = True

    trainer = Trainer(args.device, args.generator_type, args.discriminator_type, args.learning_rate)
    dataloader = Reader(args.data_root)
    trainer.train_loop(dataloader, args.epochs, args.img_size, args.noise_size, args.batch_size, 3, args.proportion, log)
