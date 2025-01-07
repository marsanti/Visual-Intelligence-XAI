import os
from torch.utils.data import Dataset
from utils.config import *
from PIL import Image

class VisualDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx][0]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

def get_imgs_list(dir_path):
    imgs = []
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            imgs.append((img_path, class_name))

    labels = []
    for img in imgs:
        if img[1] == 'adenocarcinoma':
            labels.append(0)
        else:
            labels.append(1)

    return imgs, labels