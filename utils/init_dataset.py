from config import *
import gdown
import os
import zipfile
import time

def init_dataset():
    start = time.time()
    # download the zip for adenocarcinoma and benign classes from google drive
    # create path string 
    adeno_path = os.path.join(DATASET_PATH, 'adenocarcinoma.zip')
    benign_path = os.path.join(DATASET_PATH, 'benign.zip')
    # download the files
    gdown.download(id=ADENOCARCINOMA_GD_ID, output=adeno_path)
    gdown.download(id=BENIGN_GD_ID, output=benign_path)
    # unzip the files
    print('Unzipping files...')
    with zipfile.ZipFile(adeno_path, 'r') as zipf:
        zipf.extractall(os.path.join(DATASET_PATH, 'adenocarcinoma'))

    with zipfile.ZipFile(benign_path, 'r') as zipf:
        zipf.extractall(os.path.join(DATASET_PATH, 'benign'))
    # clean up
    # Check if the file exists and delete it
    print('Cleaning up...')
    if os.path.exists(adeno_path):
        os.remove(adeno_path)
    if os.path.exists(benign_path):
        os.remove(benign_path)
    print(f'Done in {time.time() - start:.2f} seconds')
if __name__ == '__main__':
    init_dataset()