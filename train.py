import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from chess_neural_network import ChessNN

# Convolutional Neural Network
cnn = ChessNN()

# Parameters
BATCH_SIZE = 4
DATA_DIR = 'dataset/'
NUM_WORKERS = 0 # If running on Windows and get a BrokenPipeError, try setting NUM_WORKERS=0
EPOCHS = 10
LEARNING_RATE = 0.001

# Transforms
transform = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training image dataset
trainset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Testing image dataset
testset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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

# === Save the model ===
torch.save(cnn.state_dict(), 'model.pth')

print("Finished Training")
