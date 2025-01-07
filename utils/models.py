from torch import nn, softmax
from sklearn.svm import SVC
from kymatio.torch import Scattering2D

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

class ScatNet(nn.Module):
    def __init__(self, J: int = 3, L: int = 8, input_shape: tuple = (224, 224)):
        super(ScatNet, self).__init__()
        self.scatLayer = Scattering2D(J=J, L=L, shape=input_shape)
        input_classifier_shape = self._calc_input_classifier_shape(J, L, input_shape)
        print(f'Input classifier shape: {input_classifier_shape}')
        self.classifier = Classifier(input_shape=input_classifier_shape, output_shape=1)

    def forward(self, x):
        x = x.squeeze(1) # remove the channel dimension
        # generate scattering coefficients
        x = self.scatLayer.scattering(x) # returns a tensor of shape (batch_size, coeffs, H, W)
        # print(f'Before flattening: {x.shape}')
        # flatten the scattering coefficients
        x = x.view(x.size(0), -1)
        # use the scattering coefficients as predictors for the classifier
        x = self.classifier(x)
        return x
    
    def _calc_input_classifier_shape(self, J: int, L: int, input_shape: tuple) -> int:
        """
        Calculate the input shape for the classifier layer.
        
        Returns:
            int: the input shape for the classifier layer.

        """
        n_coeffs = self._calc_n_coeffs(J, L)
        spatial_size = [input_shape[0]/(2**J), input_shape[1]/(2**J)]
        return int(n_coeffs * spatial_size[0] * spatial_size[1])

    def _calc_n_coeffs(self, J: int, L: int) -> int:
        """
        Calculate the number of coefficients for the scattering network.

        source: https://www.kymat.io/userguide.html?utm_source=chatgpt.com#id13
        
        Returns:
            int: number of coefficients calculated by the formula:
                        1 + J*L + ((J*(J-1))/2)*(L**2)
                 where:
                        1: the number of zeroth order coefficients
                        J*L: number of first order coefficients
                        ((J*(J-1))/2)*(L**2): number of second order coefficients

        """
        return int(1 + (J*L) + (((J*(J-1))/2)*(L**2)))