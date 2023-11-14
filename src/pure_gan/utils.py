""" 
Utils file
"""
from pathlib import Path
import logging
import string
import random

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def check_and_create_dir(directory):
    """Check and create dir

    Args:
        dir (str): directory
    """
    path = Path(f"{directory}/")
    if path.exists():
        print(f"The path '{path}' exists.")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"The path '{path}' created successfully.")


def get_random_string(length):
    """Get random string as prefix of name of the model

    Args:
        length (int): length of string

    Returns:
        str: A random string
    """
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str
