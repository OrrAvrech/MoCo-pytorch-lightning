# MoCo - Pytorch Lightning

In this assignment, we designed a self-supervised framework, based on the MoCo-V1 
[1] and MoCo-V2 [2] papers. We started with an unsupervised training on the *Imagenette* dataset. 
Then, we used the extracted features from the unsupervised training step and trained a classifier in a fully supervised manner,
and compared to an **ImageNet** pre-trained feature extractor. 

## Setup
You can create the project's environment from the `env/env.yml` file by running:
```
> cd env
> conda env create -f env.yml
> pip install -r main.txt
```

## Run
- Set parameters for training in `params.py`
- Train MoCo on *Imagenette* in `train_moco.py`
- Apply linear evaluation on *Imagenette* in `train_linear_classifier.py`

## References
[1] [Momentum Contrast for Unsupervised Visual Representation Learning, He et al., 2020](https://arxiv.org/pdf/1911.05722.pdf)

[2] [Improved Baselines with Momentum Contrastive Learning, Chen et al., 2020](https://arxiv.org/pdf/2003.04297.pdf)
 