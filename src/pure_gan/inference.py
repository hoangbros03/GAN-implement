import argparse

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models import Generator
from utils import check_and_create_dir


def inference(model_file, latent_dim, img_shape, output_file):
    """_summary_

    Args:
        model_file (_type_): _description_
        output_file (_type_): _description_
    """
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Generator(latent_dim, img_shape).to(device)
    model.load_state_dict(torch.load(model_file))

    # Get result
    model.eval()
    sample = torch.randint(0,2,(1,latent_dim)).float().to(device)
    img = model.forward(sample)

    # Export the image
    denormalize = transforms.Normalize((-0.5/0.5), (1.0/0.5))
    img = denormalize(img)
    img = transforms.ToPILImage()(img)
    img.save(output_file)
    

if __name__=="__main__":
    check_and_create_dir("gen_images")
    inference("models/demo.pth", 100, (1,28,28), "gen_images/test.png")