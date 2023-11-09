""" 
File to get generated images
"""
import argparse

import numpy as np
import torch
from PIL import Image
from models import Generator
from utils import check_and_create_dir


def inference(model_file, latent_dim, img_shape, device, amount, image_path):
    """Inference function to get generated image

    Args:
        model_file (_type_): path of model
        latent_dim: Dimension of latent space
        img_shape: Shape of image
        device (str): cpu or cuda
        amount (int): Number of images that you want to generate
        image_path (str): Folder containing generated images
    """
    # Load model
    model = Generator(latent_dim, img_shape).to(device)
    model.load_state_dict(torch.load(model_file))

    # Get result
    model.eval()
    sample = torch.randn(amount, latent_dim) * 1.0
    img = model.forward(sample)
    print(img.shape)

    # Export the image
    img *= 255
    img = img.cpu().detach().numpy().astype(np.uint8)
    for i in range(amount):
        this_img = Image.fromarray(img[i])
        this_img.save(f"{image_path}/{model_file.split('/')[-1][:-4]}_{i}.png")


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
