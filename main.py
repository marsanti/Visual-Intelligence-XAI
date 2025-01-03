from dataset import VisualDataset, get_imgs_list
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from utils.config import *
from utils.models import *
from utils.utils import train_model

def main():
    X, Y = get_imgs_list(DATASET_PATH)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    train_dataset = VisualDataset(X_train, Y_train, transform=TRANSFORM)
    test_dataset = VisualDataset(X_test, Y_test, transform=TRANSFORM)
    
    # create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    
    # instantiate models
    cnn = CNN(input_shape=3, output_shape=1).to(DEVICE)
    scatnet = ScatNet()

    # define loss and optimizer for the CNN
    cnn_loss_fn = nn.BCEWithLogitsLoss()
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train_model(model=cnn, loss_fn=cnn_loss_fn, optimizer=cnn_optimizer, data_loader=train_data_loader)

if __name__ == '__main__':
    main()