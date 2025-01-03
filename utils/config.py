from torchvision import transforms

# Dataset configuration
DATASET_PATH = 'dataset'
ADENOCARCINOMA_GD_ID = '1xJ18HbCb4mWgObZuOPE5gNNKwUixAcZk'
BENIGN_GD_ID = '15fOw0aNfQ8meXZH_sOrbinaUetsoFTS0'
TRANSFORM = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor()
])

RANDOM_STATE = 42
BATCH_SIZE = 32