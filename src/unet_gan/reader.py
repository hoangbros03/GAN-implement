import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

class Reader:
    def __init__(self, dataroot, batch_size, workers, image_size):
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size

    def path_to_dataloader(self):
        dataset = dset.ImageFolder(root=self.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor()
                                   ]))
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size,
                                                 shuffle=True, 
                                                 num_workers=self.workers)
        return dataloader
    
    def plot(self, device):
        dataloader = self.path_to_dataloader()
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

if __name__ == "__main__":
    dataroot = "data\sketches_png\png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = Reader(dataroot=dataroot, batch_size=32, workers=2, image_size=256)
    dataloader = reader.path_to_dataloader()