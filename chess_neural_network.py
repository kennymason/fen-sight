# chess_neural_network.py
# Kenneth Mason
# Convolutional Neural Network for classifying chess pieces

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defines a Convolutional Neural Network (CNN) for chess piece recognition
class ChessNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 5) # 3 input channels (RGB), 32 output feature maps, kernel size 5
    self.pool = nn.MaxPool2d(2, 2) # kernel size 2, stride 2 (same as kernel size, as is default)
    self.conv2 = nn.Conv2d(32, 64, 5) # 32 input channels, 64 output channels
    # Output = (62 - 5 + 0)/1 + 1 = 58 -> (58x58)
    self.fc1 = nn.Linear(64 * 29 * 29, 512) # 64*29*29 size of vector after torch.flatten(), I chose 512 as output size
    self.fc2 = nn.Linear(512, 128) # 512 input size, 128 output size
    self.fc3 = nn.Linear(128, 26) # 128 input size, 26 output size (we have 26 classes)

  def forward(self, x):
    # Input Size = 128 -> (128px x 128px image)
    x = self.pool(F.relu(self.conv1(x)))
    # Conv Output Size = ((input_size - kernel_size + 2*padding) / stride) + 1 = (128 - 5 + 0)/1 + 1 = 124 -> (124x124)
    # Pool Output Size = input_size / kernel_size = 62 -> (124x124 / 2x2 = 62x62)
    x = self.pool(F.relu(self.conv2(x)))
    # Conv Output Size = (62 - 5 + 0)/1 + 1 = 58 -> (58x58)
    # Pool Output Size = 29 -> (29x29)
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    # Output Size = channels * height * width = 64 * 29 * 29 = 53,824
    x = F.relu(self.fc1(x))
    # Output Size = 512
    x = F.relu(self.fc2(x))
    # Output Size = 128
    x = self.fc3(x)
    # Output Size = 26 (26 classes)
    return x
