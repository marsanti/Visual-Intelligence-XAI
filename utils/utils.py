import os
from utils.config import *

def parse_data():
    adeno_path = os.path.join(DATASET_PATH, 'adenocarcinoma')
    benign_path = os.path.join(DATASET_PATH, 'benign')

    # list the images in the 2 classes
    # adenocarcinoma class = 0, benign class = 1
    adeno_images = os.listdir(adeno_path)
    benign_images = os.listdir(benign_path)

    # concatenate the 2 lists in order to have a single list of images
    images = adeno_images + benign_images

    labels = [0 for _ in adeno_images] + [1 for _ in benign_images]
    
    return images, labels