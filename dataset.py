import os
from torch.utils.data import Dataset
from utils.config import *
from PIL import Image

class VisualDataset(Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.adeno_path = os.path.join(DATASET_PATH, 'adenocarcinoma')
        self.benign_path = os.path.join(DATASET_PATH, 'benign')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        if label == 0:
            img_path = os.path.join(self.adeno_path, img_path)
        else:
            img_path = os.path.join(self.benign_path, img_path)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label