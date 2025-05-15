# normal_params.py
# Kenneth Mason
# Calculates the normalization parameters of a dataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load training dataset without normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # No normalization here
])

dataset = datasets.ImageFolder("dataset/train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

mean = 0.
std = 0.
num_samples = 0.

for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # flatten H and W: (batch_size, 3, H, W) to (batch_size, 3, H*W)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    num_samples += batch_samples

mean /= num_samples
std /= num_samples

print("Mean:", mean)
print("Std:", std)
