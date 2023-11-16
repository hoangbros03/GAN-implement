import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

def convert_dataloader_to_sample(dataloader):
    # batch size = 128
    for i, batch_data in enumerate(dataloader):
        if i < 4:
            if i == 0:
                gt = batch_data[0]
            else: 
                gt = torch.vstack((gt, batch_data[0]))

    return gt

def unnormalize(tensor):
    return (tensor * 255).type(torch.uint8)

def convert_rgb(tensor):
    return tensor.repeat(1, 3, 1, 1)

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