# train.py
# Kenneth Mason
# Trains the "Chess Neural Network" CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from chess_neural_network import ChessNN

# Parameters
BATCH_SIZE = 32
DATA_DIR = 'dataset/train'
NUM_WORKERS = 0 # Setting NUM_WORKERS higher than 0 will cause errors on some machines
EPOCHS = 10
LEARNING_RATE = 0.001

# Convolutional Neural Network
cnn = ChessNN()

# Transforms
transform = transforms.Compose([
  transforms.Resize((128, 128)), # Resize the image to something nice
  transforms.ToTensor(), # Converts image to a PyTorch tensor
  transforms.Normalize((0.3954, 0.3891, 0.3873), (0.2244, 0.2218, 0.2180)) # Normalize using calculated parameters
])

# Load the training image dataset
trainset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Loop
for epoch in range(EPOCHS):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = cnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 100 == 99: # print every 2000 mini-batches
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
      running_loss = 0.0

# Save the model
torch.save(cnn.state_dict(), 'model.pth')

print("Finished Training")
