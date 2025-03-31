# Intro
This project aims to compare a CNN and a ScatNet used as feature extraction models, and with that features execute a binary classification over lung cancer images.

After training the model two explainability algorithms are used to understand if the models are focusing on correct stuff.

The xAI algorithms applied are:
1. Guided Backpropagation (GBP);
2. Guided Grad-CAM.

# Setup
## Requirements
In order to duplicate our results, create an 
environment (conda or virtual) and install the 
packages findable in `requirements.txt`.

## Dataset setup
Run `utils/init_dataset.py` from the root project folder:
> python ./utils/init_dataset.py

## Config.py
Inside `utils/config.py` you can find some flags:
- __SKIP_TRAINING__: whether train the model or skip if already done;
- __SKIP_FILTER_EXTRACTION__: choose to skip the filter extraction;
- __SKIP_TESTING__: wheter testing the test set or skip it;
- __RANDOM_STATE__: set to 42 in order to reproduce our results.
- __HYPER PARAMETERS__:
    - __K_FOLDS__: number of folds to use when doing the cross validation step;
    - __EPOCHS__: number of training epochs;
    - __LEARNING_RATE__: rate by which the models are learning;
    - __BATCH_SIZE__: how many images process at a time.

Please, do ___not___ edit other config variables, if you want to reproduce the result correctly.

# Usage
To reproduce the results execute main.py after setup phase:
> python main.py

### Authors
[@marsanti](https://github.com/marsanti) and [@DanielePasotto](https://github.com/DanielePasotto)