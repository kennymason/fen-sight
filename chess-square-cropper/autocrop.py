# Kenneth Mason - 4/8/2025
# Automatic Chess Square Cropping Tool
# Given an image of a 1128 x 1128 pixel chessboard, crops and saves each square individually.
from PIL import Image
import os
import sys

# Set up paths
output_path = 'output_square_images'
# Get source image as command line argument
if len(sys.argv) < 2:
  print("Error: Path of source image required as argument")
  sys.exit(1)
board_image_path = sys.argv[1]

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
  os.makedirs(output_path)

# Full board edge length - 1128px
# Square edge length - 141px
square_edge_length = 141

# Open image with Pillow
board = Image.open(board_image_path)

# Crop individual squares one by one
num_squares = 0
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] # For naming squares
for row in range(8):
  for col in range(8):
    # Calculate coordinates
    left = col * square_edge_length
    right = left + square_edge_length
    top = row * square_edge_length
    bottom = top + square_edge_length

    # Crop secton from coordinates
    square = board.crop((left, top, right, bottom))

    # Calculate current square name
    letter = letters[num_squares % 8]
    number = 8 - int(num_squares / 8)

    # Save square image
    square.save(os.path.join(output_path, f"{letter}{number}.png"))

    num_squares += 1

print(f"{num_squares} saved to {output_path}")
