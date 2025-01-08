from scipy.fft import fft2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt
from colorsys import hls_to_rgb
import os
import numpy as np
from tqdm import tqdm
from utils.config import *
from utils.models import CNN, ScatNet, reset_weights

def extract_conv_filters(model: CNN) -> list[torch.Tensor]:
    """
        Extract convolutional filters from a given model.

        Returns a list of tensors containing the filters.
    """
    model_weights =[]
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
        elif type(model_children[i]) == nn.Sequential:
            # inside sequential there are conv layers
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.Conv2d:
                    counter+=1
                    model_weights.append(model_children[i][j].weight)

    return model_weights

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, path_to_save=None):
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor.detach().cpu(), nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        if path_to_save:
            plt.savefig(path_to_save)
        else:
            plt.show()

def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c

def extract_and_visualize_scat_filters(model: ScatNet, path_to_save: str) -> None:
    """
        Extract scattering filters from a given model.

        Returns a list of tensors containing the filters.
    """
    J, L = model.J, model.L
    # load the filters
    filters = model.scatLayer.psi
    # plot the filters
    fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    i = 0
    for filter in filters:
        f = filter["levels"][0]
        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        axs[i // L, i % L].imshow(colorize(filter_c))
        axs[i // L, i % L].axis('off')
        axs[i // L, i % L].set_title(
            "j={}\ntheta={}".format(i // L, i % L))
        i = i+1

    fig.suptitle((r"Wavelets for each scales j and angles theta used."
                "\nColor saturation and color hue respectively denote complex "
                "magnitude and complex phase."), fontsize=13)
    plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save)
    else:
        plt.show()

def k_fold_cross_validation_train(model: any, loss_fn: any, optimizer:any, train_dataset: Dataset) -> tuple[float, float]:
    """
        Perform k-fold cross validation training for both CNN and ScatNet models.

        Returns a tuple containing:
            - mean accuracy over all folds;
            - mean F1 score over all folds.
    """
    if type(model) == CNN:
        model_name = 'CNN'
    elif type(model) == ScatNet:
        model_name = 'ScatNet'
    else:
        raise ValueError("Invalid model type: Available models are CNN and ScatNet.")

    model_path = os.path.join(MODEL_PATH, f'Best_{model_name}.pth')

    k_fold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    acc_mean = 0.0
    f1_mean = 0.0
    best_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_dataset)):
        print(f"\n### Fold: {fold+1}/{K_FOLDS} ###\n")

        # in order to divide the dataset into train and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)

        model.apply(reset_weights)

        loss_history, train_acc_history, f1_score_history, val_acc = train_model(model, loss_fn, optimizer, train_dataloader, val_dataloader)
        
        acc_mean += val_acc

        f1_mean += np.mean(f1_score_history)

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

    return (acc_mean/K_FOLDS, f1_mean/K_FOLDS)

def train_model(model: any, loss_fn: any, optimizer: any, train_data_loader: DataLoader, val_data_loader: DataLoader = None) -> tuple[list[torch.Tensor], list[float], list[float], float]:
    """
        Utility function to train a given model.

        Returns a tuple containing:
            - loss_history: list of loss values for training;
            - training_acc_list: list of accuracy values for training;
            - f1_score_list: list of F1 scores for training;
            - validation_acc: accuracy value for validation.
    """
    train_loss_history = []
    train_acc_history = []
    val_acc = 0.0
    f1_history = []

    # training loop
    for epoch in range(EPOCHS):
        # set the model to train mode
        model.train()
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        pbar.set_description(f'Epoch 1/{EPOCHS} | train | Loss: 0.0000 | Acc: 0.00%')
        running_loss = 0.0
        acc_score = 0.0
        f1 = 0.0
        for i, (X, Y) in pbar:
            X = X.type(torch.float).to(DEVICE)
            Y = Y.type(torch.float).to(DEVICE)
            optimizer.zero_grad()
            Y_pred = model(X).squeeze()
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss_history.append(loss.item())
            acc_score += accuracy_score(Y.cpu().detach().numpy(), torch.round(torch.sigmoid(Y_pred)).cpu().detach().numpy())
            f1 += f1_score(Y.cpu().detach().numpy(), torch.round(torch.sigmoid(Y_pred)).cpu().detach().numpy())

            pbar.set_description(
                    f'Epoch {epoch+1}/{EPOCHS} | train | Loss: {running_loss/(i+1):.4f} | Acc: {(acc_score/((i+1)))*100:.2f}%')
        
        f1_history.append(f1/len(train_data_loader))
        train_acc_history.append(acc_score/len(train_data_loader))

    if val_data_loader:
        val_acc =  test_model(model, loss_fn, val_data_loader, is_val_phase=True)

    return (train_loss_history, train_acc_history, f1_history, val_acc)
            
def test_model(model: any, loss_fn: any, data_loader: DataLoader, is_val_phase: bool = False) -> torch.Tensor:
    """
        Utility function to test a given model.

        Returns the accuracy value for the given model.
    """
    model.eval()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    phase = 'val' if is_val_phase else 'test'
    pbar.set_description(f'{phase} | Loss: 0.0000 | Acc: 0.00%')
    running_loss = 0.0
    acc_score = 0
    # testing loop
    for i, (X, Y) in pbar:
        X = X.type(torch.float).to(DEVICE)
        Y = Y.type(torch.float).to(DEVICE)
        Y_pred = model(X).squeeze()
        loss = loss_fn(Y_pred, Y)
        
        running_loss += loss.item()
        acc_score += accuracy_score(Y.cpu().detach().numpy(), torch.round(torch.sigmoid(Y_pred)).cpu().detach().numpy())

        pbar.set_description(
                f'{phase} | Loss: {running_loss/(i+1):.4f} | Acc: {(acc_score/(i+1))*100:.2f}%')
    
    return acc_score