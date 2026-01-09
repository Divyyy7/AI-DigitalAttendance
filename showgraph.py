import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# -----------------------------
# Paths
# -----------------------------
EMBEDDING_DIR = r"C:\Users\HP\OneDrive\cvproject\embeddings"

# -----------------------------
# Load embeddings
# -----------------------------
X = []
labels = []

for file in os.listdir(EMBEDDING_DIR):
    if file.endswith(".pt"):
        path = os.path.join(EMBEDDING_DIR, file)

        emb = torch.load(path)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy().flatten()
        else:
            emb = emb.flatten()

        name = file.split("_")[0]  # full_name
        X.append(emb)
        labels.append(name)

X = np.array(X)

# Normalize (important)
X = normalize(X, norm="l2")

print("Total embeddings:", len(X))

# -----------------------------
# Reduce to 2D using PCA
# -----------------------------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print("Explained variance:", pca.explained_variance_ratio_)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 8))

color_map = {}
colors = []
for i in range(30):
    hue = (i * 360 / 30) / 360  # even spacing in hue
    color = mcolors.hsv_to_rgb((hue, 1.0, 1.0))  # full saturation and value for brightness
    colors.append(color)
label_index = 0

for i, name in enumerate(labels):
    if name not in color_map:
        color_map[name] = colors[label_index % len(colors)]
        label_index += 1

    plt.scatter(
        X_2d[i, 0],
        X_2d[i, 1],
        color=color_map[name],
        s=60,
        alpha=0.8
    )

# Legend
for name, color in color_map.items():
    plt.scatter([], [], color=color, label=name)

plt.legend(title="Students", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.title("Face Embeddings Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()
