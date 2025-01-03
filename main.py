from dataset import VisualDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import *
from utils.models import *
from utils.utils import parse_data

def main():
    X, Y = parse_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

    train_dataset = VisualDataset(X_train, Y_train, transform=TRANSFORM)
    test_dataset = VisualDataset(X_test, Y_test, transform=TRANSFORM)
    
    # create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    
    cnn = CNN(input_shape=3)
    scatnet = ScatNet()

    for images, labels in tqdm(train_data_loader):
        
        tqdm.update(1)

if __name__ == '__main__':
    main()