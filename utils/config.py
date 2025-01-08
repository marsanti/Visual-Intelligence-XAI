from torchvision import transforms
import torch

# Dataset configuration
DATASET_PATH = 'dataset'
# IDs from Google Drive
ADENOCARCINOMA_GD_ID = '1xJ18HbCb4mWgObZuOPE5gNNKwUixAcZk'
BENIGN_GD_ID = '15fOw0aNfQ8meXZH_sOrbinaUetsoFTS0'
# Transform images resizing them, converting them to tensors and visualizing in greyscale
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Grayscale()
])
MODEL_PATH = 'model_weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_FOLDS = 5
SKIP_TRAINING = True
EPOCHS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001