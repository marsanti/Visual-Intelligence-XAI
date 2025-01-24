import warnings
import PIL
import captum
import os
import numpy as np
import cv2
from torch import nn
from matplotlib import pyplot as plt

from utils.config import *
from utils.utils import normalize, hide_axes, get_classname
from XAI_methods.guided_backpropagation import custom_guided_backpropagation

class custom_guided_gradCAM():
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.guided_backprop = custom_guided_backpropagation(model)

        # register hooks in order to calculate gradCAM
        self.hooks = []
        self.gradients = None
        self.activations = None

        # Register hooks for gradients and activations
        self.hooks.append(target_layer.register_forward_hook(self.forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(self.full_backward_hook))

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def full_backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _get_backprop_attr(self, image):
        return self.guided_backprop.visualize(image)

    def _compute_heatmap(self, input_batch):
        """
            Compute the Grad-CAM heatmap for a given input batch.

            Parameters:
                input_batch: torch.Tensor
                    The input batch to compute the heatmap for

            Returns:
                heatmap: np.ndarray
                    The Grad-CAM heatmap
        """
        # Forward pass
        logits = self.model(input_batch)
        self.model.zero_grad()

        # step 1: compute the gradients
        logits.backward()

        # Compute Grad-CAM heatmap
        # step 2: compute the GAP (Global Average Pool) of feature map
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # step 3: compute the final Grad-CAM localization map
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        # Normalize to [0, 1]
        heatmap /= torch.max(heatmap)

        return heatmap.squeeze().cpu().numpy()

    def run(self, image):
        """
            This function is used to compute the guided Grad-CAM attributions for a given image.

            Parameters:
                image: torch.Tensor
                    The image to compute the attributions for

            Returns:
                cam_gb: np.ndarray
                    The guided Grad-CAM attributions
        """
        guided_backprop_attr = self._get_backprop_attr(image)
        heatmap = self._compute_heatmap(image)

        # resize heatmap to the image size
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
        # transform the heatmap to int to apply color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # get the heatmap back to float
        heatmap = heatmap.astype(np.float32) / 255

        # multiply the heatmap with the guided backpropagation attributions element-wise
        cam_gb = np.multiply(heatmap, guided_backprop_attr)

        return cam_gb

def demo_guided_gradCAM(model, adeno_image = None, benign_image = None):
    """
        This function is used to visualize the guided Grad-CAM attributions for an adenocarcinoma and a benign image.\n
        It's a demo function, to show the difference between the custom implementation and the captum implementation.

        Parameters:
            model: nn.Module
                The model to be used for the visualization
            adeno_image: torch.Tensor
                The adenocarcinoma image to visualize
            benign_image: torch.Tensor
                The benign image to visualize
    """
    # open an adenocarcinoma image and a benign image if no images were given and transform the images
    if adeno_image is None:
        adeno_image = PIL.Image.open('dataset/adenocarcinoma/4999.jpg')
        adeno_image = TRANSFORM(adeno_image).to(DEVICE)
    if benign_image is None:
        benign_image = PIL.Image.open('dataset/benign/4999.jpg')
        benign_image = TRANSFORM(benign_image).to(DEVICE)
    # add batch dimension and set requires_grad to True
    adeno_image_for_attr = adeno_image.unsqueeze(0).requires_grad_()
    benign_image_for_attr = benign_image.unsqueeze(0).requires_grad_()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        custom_GGCAM = custom_guided_gradCAM(model, model.layer_4[3])
        adeno_custom_attr = custom_GGCAM.run(adeno_image_for_attr)
        benign_custom_attr = custom_GGCAM.run(benign_image_for_attr)

    # execute guided backprop on each image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.eval()
        captum_gbp = captum.attr.GuidedGradCam(model, model.layer_4[3])
        adeno_captum_attr = captum_gbp.attribute(adeno_image_for_attr, target=None)
        benign_captum_attr = captum_gbp.attribute(benign_image_for_attr, target=None)

    # normalize the attributions
    adeno_custom_attr = normalize(adeno_custom_attr)
    benign_custom_attr = normalize(benign_custom_attr)

    adeno_captum_attr = normalize(adeno_captum_attr)
    benign_captum_attr = normalize(benign_captum_attr)

    # predict classes
    predicted_adeno_class = get_classname(torch.round(torch.sigmoid(model(adeno_image.unsqueeze(0)))))
    predicted_benign_class = get_classname(torch.round(torch.sigmoid(model(benign_image.unsqueeze(0)))))

    # visualize
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(adeno_image.cpu().permute(1,2,0).squeeze())
    ax[0, 0].set_title('Original Adenocarcinoma Image')
    ax[0, 0].set_xlabel(f'Predicted class: {predicted_adeno_class}')
    hide_axes(ax[0, 0])
    ax[0, 1].imshow(adeno_custom_attr)
    ax[0, 1].set_title('Adenocarcinoma Custom \nGuided Grad-CAM')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(adeno_captum_attr.squeeze().detach().cpu().permute(1,2,0))
    ax[0, 2].set_title('Adenocarcinoma Captum \nGuided Grad-CAM')
    ax[0, 2].axis('off')
    ax[1, 0].imshow(benign_image.cpu().permute(1,2,0).squeeze())
    ax[1, 0].set_title('Original Benign Image')
    ax[1, 0].set_xlabel(f'Predicted class: {predicted_benign_class}')
    hide_axes(ax[1, 0])
    ax[1, 1].imshow(benign_custom_attr)
    ax[1, 1].set_title('Benign Custom \nGuided Grad-CAM')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(benign_captum_attr.squeeze().detach().cpu().permute(1,2,0))
    ax[1, 2].set_title('Benign Captum \nGuided Grad-CAM')
    ax[1, 2].axis('off')
    plt.tight_layout()
    path = os.path.join(XAI_RESULTS_PATH, 'guided_gradCAM.png')
    plt.savefig(path)