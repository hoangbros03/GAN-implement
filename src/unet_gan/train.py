import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from layers import Discriminator, Generator

class Trainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.criterion = nn.BCELoss()
        self.netG = Generator().to(device)
        self.netD = Discriminator().to(device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

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

                # print("Generating on fixed noise...")
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1