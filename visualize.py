# visualize.py
# Kenneth Mason
# Visualize extracted features with PCA and LDA

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from config import TRAIN_DATA_DIR

# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')

# Normalize features before applying PCA/LDA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Get dict of class names
dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR)
class_names = dataset.classes

# Custom legend
cmap = plt.cm.get_cmap('tab20', len(class_names)) # Colormap
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=class_names[i],
      markerfacecolor=cmap(i), markersize=6)
    for i in np.unique(labels)
]

## PCA ##
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='tab20', alpha=0.7, s=10)
plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("PCA Projection (2D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()

## LDA ##
# LDA can use at most C-1 components, where C is number of classes
num_classes = len(np.unique(labels))
lda = LDA(n_components=min(num_classes - 1, 2))
features_lda = lda.fit_transform(features_scaled, labels)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_lda[:, 0], features_lda[:, 1], c=labels, cmap='tab20', alpha=0.7, s=10)
plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("LDA Projection (2D)")
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.tight_layout()
plt.show()
