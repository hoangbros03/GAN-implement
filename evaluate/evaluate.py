"""File to evaluate"""
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance


def convert_dataloader_to_sample(dataloader):
    """Get sample from dataloader

    Args:
        dataloader (): The dataloader (MNIST or CelebA)

    Returns:
        torch.Tensor: Some samples of images, for example [128,1,28,28]
        if loaded from MNIST dataset
    """
    # batch size = 128
    for i, batch_data in enumerate(dataloader):
        if i < 4:
            if i == 0:
                gt = batch_data[0]
            else:
                gt = torch.vstack((gt, batch_data[0]))
        else:
            return gt
    return None


def unnormalize(tensor):
    """Convert each value inside tensor into normal pixel value

    Args:
        tensor (torch.Tensor): tensor

    Returns:
        tensor (torch.Tensor): tensor but unnormalized
    """
    return (tensor * 255).type(torch.uint8)


def convert_rgb(tensor):
    """From grayscale -> RGB (for MNIST exclusively)

    Args:
        tensor (torch.Tensor): tensor

    Returns:
        tensor (torch.Tensor): tensor but RGB
    """
    return tensor.repeat(1, 3, 1, 1)


def kernel_inception_distance(prediction, gt):
    """Calculate the kernel inception distance

    Args:
        prediction (torch.Tensor): generated images
        gt (torch.Tensor): real images

    Returns:
        A number indicate the value from the specified metric
    """
    kid = KernelInceptionDistance(subset_size=64)
    kid.update(prediction, real=False)
    kid.update(gt, real=True)
    return kid.compute()


def frechet_inception_distance(prediction, gt):
    """Calculate the frechet_inception_distance

    Args:
        prediction (torch.Tensor): generated images
        gt (torch.Tensor): real images

    Returns:
        A number indicate the value from the specified metric
    """
    fid = FrechetInceptionDistance(feature=64)
    fid.update(prediction, real=False)
    fid.update(gt, real=True)
    return fid.compute()
