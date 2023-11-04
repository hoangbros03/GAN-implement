from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import check_and_create_dir

def check_and_process_dataloader(dir, img_shape,batch_size):
    """_summary_ TODO

    Args:
        batch_size (_type_): _description_
    """
    check_and_create_dir(dir)
    transform = transforms.Compose(
        [
            transforms.Resize(img_shape[1], img_shape[2]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return DataLoader(
        datasets.MNIST(
            f"./{dir}/",
            train=True,
            download=True,
            transform=transform,
            batch_size = batch_size,
            shuffle=True
        )
    )