# test.py
# Kenneth Mason
# Tests the trained model, with optional image augmentations

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from chess_neural_network import ChessNN
from config import TEST_DATA_DIR, MODEL_PATH, MEAN, STD, RESIZE_DIM, BATCH_SIZE, NUM_WORKERS

# Load the model
cnn = ChessNN()
cnn.load_state_dict(torch.load(MODEL_PATH))
cnn.eval()  # Set the model to evaluation mode

# Transforms
transform = transforms.Compose([
  transforms.Resize(RESIZE_DIM), # Resize the image to something nice
  # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # Slight color shifts
  # transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images
  # transforms.RandomRotation(15), # Random small rotation (-15 to +15 degrees)
  # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Small random shifts
  transforms.ToTensor(), # Converts the image to a PyTorch tensor
  transforms.Normalize(MEAN, STD) # Normalize
])

# Load the test dataset
testset = torchvision.datasets.ImageFolder(root=TEST_DATA_DIR, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Testing
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability

    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
    
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Build the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = testset.classes  # Automatically gets class folder names from ImageFolder

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=90, cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()

# Normalized Confusion Matrix - Shows proportions
cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp.plot(xticks_rotation=90, cmap="Blues")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

# Accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')
