import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

def convert_dataloader_to_sample(dataloader):
    for i, batch_data in enumerate(dataloader):
        if i < 16:
            if i == 0:
                gt = batch_data[0]
            else: 
                gt = torch.vstack((gt, batch_data[0]))

def kernel_inception_distance(prediction, gt):
    kid = KernelInceptionDistance(subset_size=1)
    kid.update(prediction, real=False)
    kid.update(gt, real=True)
    return kid.compute()

def frechet_inception_distance(prediction, gt):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(prediction, real=False)
    fid.update(gt, real=True)
    return fid.compute()