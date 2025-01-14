import warnings
import PIL
import captum
import os
import numpy as np
import cv2
from torch import nn
from matplotlib import pyplot as plt

from utils.config import *
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
        return self.guided_backprop.visualize(image, None)

    def _compute_heatmap(self, input_batch, class_idx=None):
        # Forward pass
        logits = self.model(input_batch)
        self.model.zero_grad()

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # Compute gradients for the target class
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0, class_idx] = 1
        logits.backward(gradient=one_hot_output)

        # Compute Grad-CAM heatmap
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)  # ReLU removes negative values
        heatmap /= torch.max(heatmap)  # Normalize to [0, 1]

        return heatmap.squeeze().cpu().numpy()

    def run(self, image):
        guided_backprop_attr = self._get_backprop_attr(image)
        heatmap = self._compute_heatmap(image)

        # resize heatmap to the image size
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap = heatmap.astype(np.float32) / 255

        cam_gb = np.multiply(heatmap, guided_backprop_attr)

        return cam_gb

def demo_guided_gradCAM(model):
    # open an adenocarcinoma image and a benign image
    adeno_image = PIL.Image.open('dataset/adenocarcinoma/0000.jpg')
    benign_image = PIL.Image.open('dataset/benign/0000.jpg')
    # transform the images
    adeno_image = TRANSFORM(adeno_image).to(DEVICE)
    benign_image = TRANSFORM(benign_image).to(DEVICE)
    # add batch dimension and set requires_grad to True
    adeno_image_for_attr = adeno_image.unsqueeze(0).requires_grad_()
    benign_image_for_attr = benign_image.unsqueeze(0).requires_grad_()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        custom_GGCAM = custom_guided_gradCAM(model, model.layer_4[3])
        adeno_custom_attr = custom_GGCAM.run(adeno_image_for_attr)
        benign_custom_attr = custom_GGCAM.run(benign_image_for_attr)

    # normalize the attributions
    adeno_custom_attr = (adeno_custom_attr - adeno_custom_attr.min()) / (adeno_custom_attr.max() - adeno_custom_attr.min())
    benign_custom_attr = (benign_custom_attr - benign_custom_attr.min()) / (benign_custom_attr.max() - benign_custom_attr.min())

    # execute guided backprop on each image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.eval()
        captum_gbp = captum.attr.GuidedGradCam(model, model.layer_4[3])
        adeno_captum_attr = captum_gbp.attribute(adeno_image_for_attr, target=None)
        benign_captum_attr = captum_gbp.attribute(benign_image_for_attr, target=None)

    # visualize
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(adeno_image.cpu().permute(1,2,0).squeeze(), cmap='gray')
    ax[0, 0].set_title('Original Adenocarcinoma Image')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(adeno_custom_attr)
    ax[0, 1].set_title('Adenocarcinoma Custom GBP')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(adeno_captum_attr.squeeze().detach().cpu().numpy())
    ax[0, 2].set_title('Adenocarcinoma Captum GBP')
    ax[0, 2].axis('off')
    ax[1, 0].imshow(benign_image.cpu().permute(1,2,0).squeeze(), cmap='gray')
    ax[1, 0].set_title('Original Benign Image')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(benign_custom_attr)
    ax[1, 1].set_title('Benign Custom GBP')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(benign_captum_attr.squeeze().detach().cpu().numpy())
    ax[1, 2].set_title('Benign Captum GBP')
    ax[1, 2].axis('off')
    plt.tight_layout()
    path = os.path.join(XAI_RESULTS_PATH, 'guided_gradCAM.png')
    plt.savefig(path)