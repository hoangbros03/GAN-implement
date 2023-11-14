import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, 
                 noise_size=100, 
                 img_size=64, 
                 channel=3):
        
        super(Generator, self).__init__()
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
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, img_size=64, channel=3):
        super(Discriminator, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Generator_Ablation(nn.Module):
    def __init__(self, 
                 noise_size=100, 
                 img_size=64, 
                 channel=3):
        
        super(Generator, self).__init__()
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
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator_Ablation(nn.Module):
    def __init__(self, img_size=64):
        super(Discriminator, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output