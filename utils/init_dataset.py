from config import *
import gdown
import os
import zipfile
import time

def create_project_structure():
    # Create the dataset directory if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(FILTERS_PATH):
        os.makedirs(FILTERS_PATH)
    if not os.path.exists(XAI_RESULTS_PATH):
        os.makedirs(XAI_RESULTS_PATH)

def init_dataset():
    start = time.time()
    create_project_structure()
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