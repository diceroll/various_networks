# Chainer Implementation of Various Networks

## Methods

- [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492)
- [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925)
- [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

## Dataset

- [The Food-101 Data Set](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

## Requirements

- chainer
- chainercv
- PIL
- tqdm
- requests
- albumentations

## Usage

### Example

`$ python train.py -g 0 -e 100 -b 64 -rt 1e -st 10e`

### Argparse

- **[-g/--gpu]** GPU ID (negative value indicates CPU)
- **[-e/--epoch]** Number of sweeps over the dataset to train
- **[-b/--batchsize]** Number of images in each mini-batch
- **[-s/--seed]** Random seed
- **[-rt/--report_trigger]** Interval for reporting (100iteration --> 100i, 10epoch --> 10e, default 1e)
- **[-st/--save_trigger]** Interval for saving the model (100iteration --> 100i, 10epoch --> 10e, default 1e)
- **[-lm/--load_model]** Path of the model object to load
- **[-lo/--load_optimizer]** Path of the optimizer object to load

## Experiment

- **Model** : SEResNet50/GCResNet50/AAResNet50
- **Optimizer** : AdamW + AMSGrad (alpha: 1e-3, weight_decay_rate: 1e-4)
- **Epoch** : 100
- **Batch Size** : 64
- **Augmentation (with albumentations)** : HorizontalFlip, PadIfNeeded, Rotate, Resize, RandomScale, RandomCrop

### Loss

![Loss](https://github.com/diceroll/various_networks/blob/images/loss.png)

### Accurracy

![Accuracy](https://github.com/diceroll/various_networks/blob/images/accuracy.png)
