import torch
from torch import nn
import warnings

from utils.config import *
from utils.utils import *
import PIL
import captum


class custom_guided_backpropagation():
    def __init__(self, model: nn.Module):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.hooks = []
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        # get all the modules from the model
        modules = []
        if type(self.model) == CNN:
            modules.append(list(self.model.layer_1.named_children()))
            modules.append(list(self.model.layer_2.named_children()))
            modules.append(list(self.model.layer_3.named_children()))
            modules.append(list(self.model.layer_4.named_children()))
        elif type(self.model) == ScatNet:
            modules.append(list(self.model.classifier.classifier.named_children()))

        for i in range(len(modules)):
            for name, module in modules[i]:
                if type(module) == nn.ReLU:
                    hook = module.register_forward_hook(forward_hook_fn)
                    hook2 = module.register_full_backward_hook(backward_hook_fn)
                    self.hooks.append(hook)
                    self.hooks.append(hook2)

        first_layer = modules[0][0][1]
        hook = first_layer.register_full_backward_hook(first_layer_hook_fn)
        self.hooks.append(hook)

    def remove(self) -> None:
        """
        Remove all registered hooks
        """
        for hook in self.hooks:
            hook.remove()

    def visualize(self, input_image):
        """
            Retrieve the guided backpropagation attributions for a given input image.

            Parameters:
                input_image: torch.Tensor
                    The input image to compute the attributions for
            
            Returns:
                result: np.ndarray
                    The guided backpropagation attributions
        """
        with torch.autograd.set_grad_enabled(True):
            model_output = self.model(input_image)
            # calculate gradients
            grads = torch.autograd.grad(torch.unbind(model_output), input_image)
        
        if type(self.model) == CNN:
            result = self.image_reconstruction.data[0]
        elif type(self.model) == ScatNet:
            result = grads[0].squeeze()
        return result.permute(1,2,0).cpu().numpy()
    
def demo_guided_backprop(model: nn.Module, adeno_image = None, benign_image = None):
    """
        This function is used to visualize the guided Backpropagation attributions for an adenocarcinoma and a benign image.\n
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
    # execute guided backprop on each image
    # captum guided backpropagation
    # use warnings to ignore the Warning: "Setting backward hooks on ReLU activations. The hooks will be removed after the attribution is finished"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.eval()
        captum_gbp = captum.attr.GuidedBackprop(model)
        adeno_captum_attr = captum_gbp.attribute(adeno_image_for_attr, target=None)
        benign_captum_attr = captum_gbp.attribute(benign_image_for_attr, target=None)

    # custom guided backpropagation
    custom_gbp = custom_guided_backpropagation(model)
    adeno_attr = custom_gbp.visualize(adeno_image_for_attr)
    benign_attr = custom_gbp.visualize(benign_image_for_attr)
    custom_gbp.remove()

    # normalize the attributions
    adeno_captum_attr = normalize(adeno_captum_attr)
    benign_captum_attr = normalize(benign_captum_attr)

    adeno_attr_norm = normalize(adeno_attr)
    benign_attr_norm = normalize(benign_attr)

    # predict classes
    predicted_adeno_class = get_classname(torch.round(torch.sigmoid(model(adeno_image.unsqueeze(0)))))
    predicted_benign_class = get_classname(torch.round(torch.sigmoid(model(benign_image.unsqueeze(0)))))

    # setup for correct visualization
    adeno_captum_attr = adeno_captum_attr.cpu().squeeze().permute(1,2,0)
    adeno_image = adeno_image.cpu().permute(1,2,0)
    benign_image = benign_image.cpu().permute(1,2,0)
    benign_captum_attr = benign_captum_attr.cpu().squeeze().permute(1,2,0)

    # visualize
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(adeno_image)
    ax[0, 0].set_title('Original Adenocarcinoma Image')
    ax[0, 0].set_xlabel(f'Predicted class: {predicted_adeno_class}')
    hide_axes(ax[0, 0])
    ax[0, 1].imshow(adeno_attr_norm)
    ax[0, 1].set_title('Adenocarcinoma Custom \nGuided Backpropagation')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(adeno_captum_attr)
    ax[0, 2].set_title('Adenocarcinoma Captum \nGuided Backpropagation')
    ax[0, 2].axis('off')
    ax[1, 0].imshow(benign_image)
    ax[1, 0].set_title('Original Benign Image')
    ax[1, 0].set_xlabel(f'Predicted class: {predicted_benign_class}')
    hide_axes(ax[1, 0])
    ax[1, 1].imshow(benign_attr_norm)
    ax[1, 1].set_title('Benign Custom \nGuided Backpropagation')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(benign_captum_attr)
    ax[1, 2].set_title('Benign Captum \nGuided Backpropagation')
    ax[1, 2].axis('off')
    plt.tight_layout()
    path = os.path.join(XAI_RESULTS_PATH, f'guided_backpropagation_{model.__class__.__name__}.png')
    plt.savefig(path)