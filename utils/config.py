from torchvision import transforms
import torch

# Dataset configuration
DATASET_PATH = 'dataset'
# IDs from Google Drive
ADENOCARCINOMA_GD_ID = '1xJ18HbCb4mWgObZuOPE5gNNKwUixAcZk'
BENIGN_GD_ID = '15fOw0aNfQ8meXZH_sOrbinaUetsoFTS0'
# Transform images resizing them, converting them to tensors
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
MODEL_PATH = 'model_weights'
FILTERS_PATH = 'filters'
XAI_RESULTS_PATH = 'XAI_methods/example_images'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_FOLDS = 5
SKIP_TRAINING = True
SKIP_FILTERS_EXTRACTION = False
SKIP_TESTING = False
EPOCHS = 10
RANDOM_STATE = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001