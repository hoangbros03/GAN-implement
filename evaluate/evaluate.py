from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

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