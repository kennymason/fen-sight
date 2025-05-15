# bhattacharrya.py
# Kenneth Mason
# Calculates the Bhattacharyya distances between class distributions

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from itertools import combinations
from config import TRAIN_DATA_DIR

# Load and scale
features = np.load('features.npy')
labels = np.load('labels.npy')
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Prepare class names
dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
class_names = dataset.classes
class_pairs = list(combinations(np.unique(labels), 2))  # all pairwise combinations

def bhattacharyya_distance(class1_data, class2_data):
  mu1 = np.mean(class1_data, axis=0)
  mu2 = np.mean(class2_data, axis=0)
  cov1 = np.cov(class1_data, rowvar=False)
  cov2 = np.cov(class2_data, rowvar=False)
  cov_avg = 0.5 * (cov1 + cov2)

  mean_diff = mu1 - mu2
  cov_avg_inv = np.linalg.pinv(cov_avg)

  term1 = 0.125 * mean_diff.T @ cov_avg_inv @ mean_diff

  # Use slogdet for better numerical stability
  sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
  sign1, logdet1 = np.linalg.slogdet(cov1)
  sign2, logdet2 = np.linalg.slogdet(cov2)

  # Check for non-positive determinants
  if sign_avg <= 0 or sign1 <= 0 or sign2 <= 0:
    # Handle singular covariance matrices gracefully, e.g. skip or assign large distance
    # or add a small regularization term (like 1e-6*I) to covariance matrices beforehand
    return np.inf  # or some large number

  term2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))

  return term1 + term2

# Function to compute all distances
def compute_distances(projected_features, labels, method_name):
  print(f"\n--- Bhattacharyya distances for {method_name} ---")
  distances = {}
  for (i, j) in class_pairs:
    data_i = projected_features[labels == i]
    data_j = projected_features[labels == j]
    dist = bhattacharyya_distance(data_i, data_j)
    distances[(i, j)] = dist
    print(f"{class_names[i]} vs {class_names[j]}: {dist:.4f}")
  return distances

# PCA projection
# Use enough components to explain 99% of variance
pca = PCA(n_components=0.99)
pca_proj = pca.fit_transform(features_scaled)
pca_distances = compute_distances(pca_proj, labels, "PCA")

# LDA projection
num_classes = len(np.unique(labels))
if num_classes >= 2:
  # Use (num_classes - 1) components for maximum dimesions
  lda = LDA(n_components=num_classes - 1)
  lda_proj = lda.fit_transform(features_scaled, labels)
  lda_distances = compute_distances(lda_proj, labels, "LDA")
else:
  print("Not enough classes for LDA")
