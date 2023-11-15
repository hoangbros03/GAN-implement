""" 
File to get generated images
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from models import Generator
from utils import check_and_create_dir

# TODO: Add inference for cnn_gan
def inference(model_file, args):
    """Inference function to get generated image

    Args:
        model_file (_type_): path of model
        latent_dim: Dimension of latent space
        img_shape: Shape of image
        device (str): cpu or cuda
        amount (int): Number of images that you want to generate
        image_path (str): Folder containing generated images
    """
    # Get params: model_type, latent_dim, img_shape, device, amount, image_path
    if args is not None:
        model_type = args.model_type
        latent_dim = args.latent_dim
        img_shape = args.img_shape
        device = args.device
        amount = args.amount
        image_path = args.image_path
    else:
        raise ValueError("Args is none!")

    if model_type=="pure_mnist":
        # Load model
        model = Generator(latent_dim, img_shape).to(device)
        model.load_state_dict(torch.load(model_file))

        # Get result
        model.eval()
        sample = torch.randn(amount, latent_dim) * 1.0
        imgs = model.forward(sample.to(device)).detach().cpu()
        print(imgs.shape)

        # Export the image
        img_to_export = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(img_to_export,(1,2,0)))
        plt.savefig(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="models", help="model file")
    parser.add_argument(
        "-ip",
        "--image_path",
        type=str,
        default="gen_images",
        help="name of folder containing generated images",
    )
    parser.add_argument(
        "-a",
        "--amount",
        type=int,
        default=5,
        help="number of images that you want to generate",
    )
    parser.add_argument("-d", "--device", type=str, default="cpu", help="cpu or cuda?")
    parser.add_argument(
        "-ld", "--latent_dim", type=int, default=100, help="Latent dim of input vector"
    )
    args = parser.parse_args()
    check_and_create_dir(args.image_path)
    inference(
        args.model,
        args.latent_dim,
        (1, 28, 28),
        args.device,
        args.amount,
        args.image_path,
    )
