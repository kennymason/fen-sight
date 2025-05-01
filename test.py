import torch
import torchvision
import torchvision.transforms as transforms
from chess_neural_network import ChessNN

# Parameters
BATCH_SIZE = 4
DATA_DIR = 'dataset/'  # Same as during training
NUM_WORKERS = 0

# Load the model
cnn = ChessNN()
cnn.load_state_dict(torch.load('model.pth'))
cnn.eval()  # Set the model to evaluation mode

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize the image to something nice
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # Slight color shifts
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images
    transforms.RandomRotation(15), # Random small rotation (-15 to +15 degrees)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Small random shifts
    transforms.ToTensor(), # Converts the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize
])

# Load the test dataset
testset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Testing
correct = 0
total = 0

# No gradient tracking needed for evaluation
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)  # get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')
