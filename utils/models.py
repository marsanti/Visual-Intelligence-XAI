from torch import nn, softmax
from sklearn.svm import SVC

from utils.config import *

class CNN(nn.Module):
    """
    Model architecture that replicate ResNet-12.
    """
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int = 64):
        super(CNN, self).__init__()

        self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        )
        self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_units*2),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*2, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden_units*2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        )
        self.layer_3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_units*4),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*4, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden_units*4),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        )
        self.layer_4 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*8, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_units*8),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*8, out_channels=hidden_units*8, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden_units*8),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = Classifier(input_shape=hidden_units*8, output_shape=output_shape)
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1) # flatten

        x = self.classifier(x)
        return x 

class Classifier(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class ScatNet():
    def __init__(self):
        self.classifier = Classifier(input_shape=..., output_shape=1)