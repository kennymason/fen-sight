# board_classification.py
# Kenneth Mason
# Takes in a cropped image of a 2D chessboard and produces its FEN record
# FEN is a standard notation for describing a chess position in ASCII text

from PIL import Image
import sys
import torch
import torchvision.transforms as transforms
from chess_neural_network import ChessNN
from config import MODEL_PATH, MEAN, STD, RESIZE_DIM, CLASS_TO_FEN

## SETUP ##

# Load the model
cnn = ChessNN()
cnn.load_state_dict(torch.load(MODEL_PATH))
cnn.eval()  # Set the model to evaluation mode

# Transforms
transform = transforms.Compose([
  transforms.Resize(RESIZE_DIM), # Resize the image to something nice
  transforms.ToTensor(), # Converts the image to a PyTorch tensor
  transforms.Normalize(MEAN, STD) # Normalize
])

# Sorted class names can be looked up using model indices
class_names = sorted(CLASS_TO_FEN.keys())

## FUNCTIONS ##

# Splits board into individual squares
def board_splitter(board):
  squares = []
  square_height, square_width = board.height // 8, board.width // 8

  for rank in range(8):
    for file in range(8):
      # Calculate square coordinates
      left = file * square_width
      right = left + square_width
      top = rank * square_height
      bottom = top + square_height

      # Isolate square and add to list
      square = board.crop((left, top, right, bottom))
      squares.append(square)

  return squares

# Classify each square as a piece (type + color) or as an empty square
def square_classifier(squares):
  class_indices = []

  for square in squares:
    # Transform image to tensor, add batch dimension
    square = square.convert('RGB')
    input_tensor = transform(square).unsqueeze(0)

    # No gradient tracking needed for evaluation
    with torch.no_grad():
      # Output tensor from model
      output_tensor = cnn(input_tensor)
      # Get max value index (classification)
      prediction = torch.argmax(output_tensor, dim=1).item()
      # Add classification index to list
      class_indices.append(prediction)

  return class_indices

def fen_generator(class_indices):
  fen_record = ""
  count = 0
  for rank in range(8):
    empty_square_count = 0
    for file in range(8):
      # Get the index corresponding to the class name of the current square
      class_index = class_indices[count]
      # Get the class name of the current square
      predicted_class = class_names[class_index]
      # Get FEN character based on class
      fen_value = CLASS_TO_FEN[predicted_class]

      # Check if square is classified as an empty square or a piece
      if fen_value == "1":
        # Empty square found
        empty_square_count += 1
      else:
        # Add count of preceding consecutive empty squares to FEN record before adding the next piece
        if empty_square_count > 0:
          fen_record += str(empty_square_count)
          empty_square_count = 0

        # Add piece FEN code to record
        fen_record += fen_value

      count += 1

    # If rank ends with empty squares, add the count to the FEN record
    if empty_square_count > 0:
      fen_record += str(empty_square_count)

    # FEN segments ranks with '/'
    if rank != 7:
      fen_record += '/'
  
  # Add additional FEN information
  # This information cannot be derived from the position alone, so we assume:
  # It's white move, no castling rights, no en passant target square, 0 halfmoves, fullmove 1
  fen_record += " w - - 0 1"

  return fen_record

## PIPELINE ##

# Step 1: Load source image, partition squares

# Get image path as command line argument
if len(sys.argv) < 2:
  print("Error: Path of source image required as argument")
  sys.exit(1)
board_image_path = sys.argv[1]

# Open image with Pillow
board = Image.open(board_image_path)

# Split into squares
squares = board_splitter(board)

# Step 2: Classify each square

# The model evaluates each square, and returns the index of the corresponding class name
class_indices = square_classifier(squares)

# Step 3: Generate FEN record

# Using the ordered class indices, get the classification of each square and produce the positional FEN record
fen_record = fen_generator(class_indices)

print(fen_record)
