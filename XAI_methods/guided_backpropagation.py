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
        modules.append(list(self.model.layer_1.named_children()))
        modules.append(list(self.model.layer_2.named_children()))
        modules.append(list(self.model.layer_3.named_children()))
        modules.append(list(self.model.layer_4.named_children()))

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

    def visualize(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        
        grad_target_map = torch.zeros((1, 1),
                                    dtype=torch.float).to(DEVICE)
        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1
        
        model_output.backward(grad_target_map)
        
        result = self.image_reconstruction.data[0].permute(1,2,0)
        return result.cpu().numpy()
    
def demo_guided_backprop(model):
    # open an adenocarcinoma image and a benign image
    adeno_image = PIL.Image.open('dataset/adenocarcinoma/0000.jpg')
    benign_image = PIL.Image.open('dataset/benign/0000.jpg')
    # transform the images
    adeno_image = TRANSFORM(adeno_image).to(DEVICE)
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
    adeno_attr = custom_gbp.visualize(adeno_image_for_attr, None)
    benign_attr = custom_gbp.visualize(benign_image_for_attr, None)
    custom_gbp.remove()

    adeno_attr_norm = normalize(adeno_attr)
    benign_attr_norm = normalize(benign_attr)

    # visualize
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(adeno_image.cpu().permute(1,2,0).squeeze(), cmap='gray')
    ax[0, 0].set_title('Original Adenocarcinoma Image')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(adeno_attr_norm, cmap='gray')
    ax[0, 1].set_title('Adenocarcinoma Custom GBP')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(adeno_captum_attr.squeeze().cpu().numpy(), cmap='gray')
    ax[0, 2].set_title('Adenocarcinoma Captum GBP')
    ax[0, 2].axis('off')
    ax[1, 0].imshow(benign_image.cpu().permute(1,2,0).squeeze(), cmap='gray')
    ax[1, 0].set_title('Original Benign Image')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(benign_attr_norm, cmap='gray')
    ax[1, 1].set_title('Benign Custom GBP')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(benign_captum_attr.squeeze().cpu().numpy(), cmap='gray')
    ax[1, 2].set_title('Benign Captum GBP')
    ax[1, 2].axis('off')
    plt.tight_layout()
    path = os.path.join(XAI_RESULTS_PATH, 'guided_backpropagation.png')
    plt.savefig(path)