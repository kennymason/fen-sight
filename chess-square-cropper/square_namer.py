# Simple Script for naming chess square images made by autocrop.py

import os

def rename_img(directory, filename, new_filename):
  current_path = os.path.join(directory, filename)
  new_path = os.path.join(directory, new_filename)
  os.rename(current_path, new_path)

directory = "/Users/mason/Develop/chess-eye/training-data/style-3/"
uid = "3"

for filename in os.listdir(directory):
  # White Pieces

  if filename == "A1.png":
    new_filename = f"white_rook_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "B1.png":
    new_filename = f"white_knight_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "C1.png":
    new_filename = f"white_bishop_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "D1.png":
    new_filename = f"white_queen_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "E1.png":
    new_filename = f"white_king_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "F1.png":
    new_filename = f"white_bishop_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "G1.png":
    new_filename = f"white_knight_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "H1.png":
    new_filename = f"white_rook_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "D3.png":
    new_filename = f"white_king_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "E3.png":
    new_filename = f"white_queen_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  # White Pawns
  
  if filename == "B2.png":
    new_filename = f"white_pawn_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "C2.png":
    new_filename = f"white_pawn_light_{uid}.png"
    rename_img(directory, filename, new_filename)

# Black Pieces

  if filename == "A8.png":
    new_filename = f"black_rook_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "B8.png":
    new_filename = f"black_knight_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "C8.png":
    new_filename = f"black_bishop_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "D8.png":
    new_filename = f"black_queen_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "E8.png":
    new_filename = f"black_king_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "F8.png":
    new_filename = f"black_bishop_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "G8.png":
    new_filename = f"black_knight_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "H8.png":
    new_filename = f"black_rook_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "D6.png":
    new_filename = f"black_king_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "E6.png":
    new_filename = f"black_queen_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  # Black Pawns
  
  if filename == "B7.png":
    new_filename = f"black_pawn_light_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "C7.png":
    new_filename = f"black_pawn_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

# Empty Squares
  
  if filename == "D4.png":
    new_filename = f"empty_dark_{uid}.png"
    rename_img(directory, filename, new_filename)

  if filename == "D5.png":
    new_filename = f"empty_light_{uid}.png"
    rename_img(directory, filename, new_filename)
