# extract_features.py
# Kenneth Mason
# Extract features from the model

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from chess_neural_network import ChessNN
import os

# Parameters
BATCH_SIZE = 32
DATA_DIR = 'dataset/train'
MODEL_PATH = 'model.pth'
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Uses GPU if available, otherwise CPU

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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.3954, 0.3891, 0.3873), (0.2244, 0.2218, 0.2180))
])

# Dataset
dataset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
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
features_np = np.vstack(all_features)
labels_np = np.hstack(all_labels)

np.save('features.npy', features_np)
np.save('labels.npy', labels_np)

print("Feature extraction complete.")
