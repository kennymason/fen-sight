# config.py
# Kenneth Mason

# Parameters
BATCH_SIZE = 32
NUM_WORKERS = 0 # Setting NUM_WORKERS higher than 0 will cause errors on some machines
EPOCHS = 10
LEARNING_RATE = 0.001

# Image Resizing Dimensions
RESIZE_DIM = (128, 128)

# Normalization Parameters - calculate from training dataset with normal_params.py
MEAN = (0.3954, 0.3891, 0.3873)
STD = (0.2244, 0.2218, 0.2180)

# Paths
TRAIN_DATA_DIR = 'dataset/train'
TEST_DATA_DIR = 'dataset/test'
MODEL_PATH = 'model.pth'

# Maps class names to FEN codes
CLASS_TO_FEN = {
  "wp": "P",
  "bp": "p",
  "wn": "N",
  "bn": "n",
  "wb": "B",
  "bb": "b",
  "wr": "R",
  "br": "r",
  "wq": "Q",
  "bq": "q",
  "wk": "K",
  "bk": "k",
  "empty": "1",
}
