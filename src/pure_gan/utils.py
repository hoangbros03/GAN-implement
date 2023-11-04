from pathlib import Path
import logging

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def check_and_create_dir(dir):
    """Check and create dir

    Args:
        dir (str): directory
    """
    path = Path(f"{dir}/")
    if path.exists():
        print(f"The path '{path}' exists.")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"The path '{path}' created successfully.")
