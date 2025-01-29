import time
from XAI_methods.guided_backpropagation import *
from XAI_methods.guided_gradCAM import *
from dataset import VisualDataset, get_imgs_list
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os

from utils.config import *
from utils.models import *
from utils.utils import *
from XAI_methods import *

def main():
    print("initializing dataset...", end=' ')
    start = time.time()
    X, Y = get_imgs_list(DATASET_PATH)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    train_dataset = VisualDataset(X_train, Y_train, transform=TRANSFORM)

    test_dataset = VisualDataset(X_test, Y_test, transform=TRANSFORM)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\t Done in: {time.time() - start:.2f} seconds")

    print("initializing models...", end=' ')
    best_cnn_path = os.path.join(MODEL_PATH, 'best_CNN.pth')
    best_scatnet_path = os.path.join(MODEL_PATH, 'best_ScatNet.pth')

    cnn = CNN(input_shape=3, output_shape=1).to(DEVICE)
    cnn_loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize models
    scatnet = ScatNet().to(DEVICE)
    scat_loss_fn = nn.BCEWithLogitsLoss()
    print(f"\t\t Done in: {time.time() - start:.2f} seconds")
    # training with k-fold cross validation
    if not SKIP_TRAINING:
        print(f'\n === Training CNN model ===')
        cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

        cnn_mean_acc, cnn_mean_f1 = k_fold_cross_validation_train(model=cnn, loss_fn=cnn_loss_fn, optimizer=cnn_optimizer, train_dataset=train_dataset)

        print(f'Training with K-Fold Cross Validation Terminated!\n Mean Accuracy: {cnn_mean_acc}, Mean F1 Score: {cnn_mean_f1}')

        print(f'\n=== Training ScatNet model ===')
        scat_optimizer = torch.optim.Adam(scatnet.parameters(), lr=LEARNING_RATE)

        scat_mean_acc, scat_mean_f1 = k_fold_cross_validation_train(model=scatnet, loss_fn=scat_loss_fn, optimizer=scat_optimizer, train_dataset=train_dataset)

        print(f'Training with K-Fold Cross Validation Terminated!\n Mean Accuracy: {scat_mean_acc}, Mean F1 Score: {scat_mean_f1}')

    # Load best model weights for each model
    print("Loading best models...", end=' ')
    start = time.time()
    if not os.path.exists(best_cnn_path) or not os.path.exists(best_scatnet_path):
        raise FileNotFoundError(f'Best models not found in {MODEL_PATH} \t try training the models first.')
    
    cnn.load_state_dict(torch.load(best_cnn_path, weights_only=False))
    scatnet.load_state_dict(torch.load(best_scatnet_path, weights_only=False))
    print(f"\t\t Done in: {time.time() - start:.2f} seconds")

    if not SKIP_FILTERS_EXTRACTION:
        print('Extracting filters...', end=' ')
        start = time.time()
        cnn_filters = extract_conv_filters(cnn)
        for i in range(len(cnn_filters)):
            filter_path = os.path.join(FILTERS_PATH, f'conv{i+1}_CNN_filters.png')
            visTensor(cnn_filters[i], ch=0, allkernels=False, path_to_save=filter_path)
        filter_path = os.path.join(FILTERS_PATH, 'scatNet_filters.png')
        extract_and_visualize_scat_filters(scatnet, path_to_save=filter_path)
        print(f"\t\t Done in: {time.time() - start:.2f} seconds")

    # get the first adenocarcinoma and benign images from test set
    adeno_image = None
    benign_image = None
    for i in range(len(test_dataset)) :
        if test_dataset[i][1] == 0 and adeno_image is None:
            adeno_image = test_dataset[i][0].to(DEVICE)
        elif test_dataset[i][1] == 1 and benign_image is None:
            benign_image = test_dataset[i][0].to(DEVICE)
        if adeno_image is not None and benign_image is not None:
            break
    
    # execute the XAI methods on the images for both models
    # CNN model
    demo_guided_backprop(cnn, adeno_image, benign_image)
    demo_guided_gradCAM(cnn, adeno_image, benign_image)
    # ScatNet model
    demo_guided_backprop(scatnet, adeno_image, benign_image)
    
    if not SKIP_TESTING:
        print(f'\n === Testing models ===')
        print('CNN model')
        test_model(model=cnn, loss_fn=cnn_loss_fn, data_loader=test_data_loader, is_val_phase=False)
        print('ScatNet model')
        test_model(model=scatnet, loss_fn=scat_loss_fn, data_loader=test_data_loader, is_val_phase=False)

if __name__ == '__main__':
    main()