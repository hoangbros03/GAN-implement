# Introduction

This repository contains our GAN implementations, reports, and experiment results for the mid-term project of INT3412E 20 in VNU-UET

# Authors

[Nguyen Binh Nguyen](github.com/nguyenrtm)

[Tran Ba Hoang](github.com/hoangbros03)

# Usage

Before using, please clone this repo, install the requirements, and install this repo as a pip package

```
git clone https://github.com/hoangbros03/GAN-implement.git
cd GAN-implement
pip install -r requirements.txt
pip install -e .
```

## Train the original GAN

To train our original GAN, it is easy just to following these commands:

```
// From the current repo dir
cd src/pure_gan
python train.py -e 200 -b 64 --lr 0.0002 --latent_dim 100 --output_model_dir models
```

Feel free to change the config specified :D

## Train the enhanced GAN

First, please make sure that celebA data is downloaded if you want to use the celebA dataset. It should has the following folder structure:

```
cnn_gan
├──data
    ├──celebA
        ├──image1.png
        ...
        ├──imagen.png
```

Then you can train with the following command:

```
// From the current repo dir
cd src/cnn_gan
python train.py -e 200 -i 64 -b 32 -n 100 -lr 0.0001 -p 3 -dt normal -gt normal -d cpu -dr data -o output_models -s
```

## Inference

Currently you can inference using the following command (Currently support original GAN only):

```
// From the current repo dir
cd src/pure_gan
python inference.py -m <relative/absolute path to model> -ip gen_images -a 16 -d cpu -ld 100 -mt pure_mnist
```

# Pre-trained model

We provide two pre-trained model:

| Type         | Dataset | Link                                                                               |
|--------------|---------|------------------------------------------------------------------------------------|
| Original GAN | MNIST   | https://drive.google.com/file/d/1ueuLtCv5VX0H_lnGIfg9FaDI-zISmOUj/view?usp=sharing |
| Enhanced GAN | CelebA  | https://drive.google.com/file/d/1JLEYVj8DmPgTFcmVDrN0I_AnAMm6WAvB/view?usp=sharing |


# License

[MIT](https://choosealicense.com/licenses/mit/)
