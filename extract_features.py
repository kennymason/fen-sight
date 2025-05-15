# extract_features.py
# Kenneth Mason
# Extract features from the model

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from chess_neural_network import ChessNN
from config import TRAIN_DATA_DIR, MODEL_PATH, MEAN, STD, RESIZE_DIM, BATCH_SIZE, NUM_WORKERS

# Uses GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load model
cnn = ChessNN()
# Load the trained weights from model.pth
cnn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cnn.to(DEVICE)
cnn.eval()

# Modify the model to stop at the 128-dimensional layer
class FeatureExtractor(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = torch.nn.Sequential(
            base_model.conv1,
            base_model.pool,
            base_model.conv2,
            base_model.pool,
            torch.nn.Flatten(),
            base_model.fc1,
            base_model.fc2  # Output is 128D here
        )

    def forward(self, x):
        return self.features(x)

extractor = FeatureExtractor(cnn).to(DEVICE)

# Transform
transform = transforms.Compose([
    transforms.Resize(RESIZE_DIM),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Dataset
dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Extract features
all_features = []
all_labels = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        features = extractor(inputs)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Stack and save
features = np.vstack(all_features)
labels = np.hstack(all_labels)

np.save('features.npy', features)
np.save('labels.npy', labels)

print("Feature extraction complete.")
