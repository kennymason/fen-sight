# visualize_3d.py
# Kenneth Mason
# Visualize PCA and LDA in 3D plots

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from torchvision import datasets

# Load features and labels
features_np = np.load('features.npy')
labels_np = np.load('labels.npy')

# Normalize features before applying PCA/LDA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_np)

# Get dict of class names
dataset = datasets.ImageFolder(root='dataset/train')
class_names = dataset.classes

# Create colormap and legend
cmap = plt.cm.get_cmap('tab20', len(class_names))
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=class_names[i],
      markerfacecolor=cmap(i), markersize=6)
    for i in range(len(class_names))
]

## PCA 3D ##
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("PCA Projection (3D)")
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels_np, cmap='tab20', s=10)
ax.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## LDA 3D ##
num_classes = len(np.unique(labels_np))
if num_classes >= 4:
    lda = LDA(n_components=3)
    lda_result = lda.fit_transform(features_scaled, labels_np)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("LDA Projection (3D)")
    ax.scatter(lda_result[:, 0], lda_result[:, 1], lda_result[:, 2], c=labels_np, cmap='tab20', s=10)
    ax.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print(f"LDA requires at least 4 classes for a 3D plot, but only {num_classes} were found.")
