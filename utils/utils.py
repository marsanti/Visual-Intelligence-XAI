import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils.config import *

def train_model(model: any, loss_fn: any, optimizer: any, data_loader: DataLoader):
    # training loop
    for epoch in range(EPOCHS):
        # set the model to train mode
        model.train()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        running_loss = 0.0
        running_corrects = 0
        for i, (X, Y) in pbar:
            X = X.type(torch.float).to(DEVICE)
            Y = Y.type(torch.float).to(DEVICE)
            optimizer.zero_grad()
            Y_pred = model(X).squeeze()
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_corrects += torch.sum(torch.round(torch.sigmoid(Y_pred)) == Y)

            pbar.set_description(
                    f'Epoch {epoch+1}/{EPOCHS} | train | Loss: {running_loss/(i+1):.4f} | Acc: {(running_corrects/((i+1)*BATCH_SIZE))*100:.4f}')