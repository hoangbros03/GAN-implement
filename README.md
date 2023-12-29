# Introduction

This repository contains our GAN (Generative Adversarial Network) implementations, reports, and experiment results for the mid-term project of INT3412E 20 in VNU-UET.

# Description

We provided two version of GAN: The original GAN with Multi Layer Perception layers and the enhanced GAN using Convolutional Neural Network layers. People can train, test, and generate the images within few commands and don't need a deep understanding of GAN.

MNIST and CelebA datasets were used and tested. Other datasets also can be used, but require a few changes in the dataloader class. We welcome any improvement to this repo.

# Authors

This repository was made by two VNU-UET students:

[Nguyen Binh Nguyen](https://github.com/nguyenrtm)

[Tran Ba Hoang](github.com/hoangbros03)


# Usage

## Installation

Before using, please clone this repo, install the requirements, and install this repo as a pip package.

```
// (Optional) Create a new conda environment
conda init
conda create -n gan_env python==3.10 -y
conda activate gan_env

// Clone and install the necessary packages
git clone https://github.com/hoangbros03/GAN-implement.git
cd GAN-implement
pip install -r requirements.txt
pip install -e .
```

## Train the original GAN

We created some scripts containing necessary configurations to train our original GAN. It is easy just to following these commands:

```
// From the current repo dir
cd src
bash scripts/train/train_pure.sh
```

Feel free to modify the config inside the file to enable wandb logging feature or changing the hyperparameters :D

## Train the enhanced GAN

First, please make sure that celebA data is downloaded if you want to use the celebA dataset (this dataset is widely available on the Internet). It should has the following folder structure:

```
cnn_gan
├──data
    ├──celebA
        ├──image1.png
        ├──image2.png
        ...
        ├──imagen.png
```

Then you can train with the following command:

```
// From the current repo dir
cd src
bash scripts/train/train_pure.sh
```

## Inference

Currently you can inference to get generated images using original GAN only. Before doing that, make sure you either has a checkpoint after the training process or download our pre-trained model mentioned below. After that, change and uncomment the line ` # --model <relative/absolute path to model> ` inside `infer_pure.sh`. After that, following these steps:

```
// From the current repo dir
cd src
bash scripts/infer/infer_pure.sh
```

# Pre-trained models

We provide two pre-trained models. You can download these models and write few commands to get the generated images:

| Type         | Dataset | Link                                                                               |
|--------------|---------|------------------------------------------------------------------------------------|
| Original GAN | MNIST   | https://drive.google.com/file/d/1ueuLtCv5VX0H_lnGIfg9FaDI-zISmOUj/view?usp=sharing |
| Enhanced GAN | CelebA  | https://drive.google.com/file/d/1JLEYVj8DmPgTFcmVDrN0I_AnAMm6WAvB/view?usp=sharing |

# Result of training

Result on MNIST dataset, evaluate by Frechet Inception Distance and Kernel Inception Distance:

| Model    | FID    | KID                       |
|----------|--------|---------------------------|
| Pure GAN | 0.0769 | mean: 0.0501, std: 0.0088 |
| CNN GAN  | 0.0290 | mean: 0.0177, std: 0.0033 |

Result on CelebA dataset:

| Model    | FID    | KID                       |
|----------|--------|---------------------------|
| Pure GAN | n/a    | n/a                       |
| CNN GAN  | 1.2611 | mean: 0.0648, std: 0.0041 |

# Demo generated images

![MNISST demo image](https://i.ibb.co/LQJS2Y1/mnist-demo-img.png)

![Celeba demo image](https://i.ibb.co/c2Ptchq/celeba-demo-img.png)

# License

[MIT](https://choosealicense.com/licenses/mit/)

# Reference

```
@misc{goodfellow2014generative,
      title={Generative Adversarial Networks}, 
      author={Ian J. Goodfellow and Jean Pouget-Abadie and Mehdi Mirza and Bing Xu and David Warde-Farley and Sherjil Ozair and Aaron Courville and Yoshua Bengio},
      year={2014},
      eprint={1406.2661},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```