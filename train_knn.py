# train_knn.py
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KNeighborsClassifier
import joblib

EMBEDDING_DIR = r"C:\Users\HP\OneDrive\cvproject\embeddings"
MODEL_PATH = r"C:\Users\HP\OneDrive\cvproject\knn_model.joblib"
LABEL_PATH = r"C:\Users\HP\OneDrive\cvproject\label_encoder.joblib"

X = []
y = []

# Load all .pt embeddings
for file in os.listdir(EMBEDDING_DIR):
    if file.endswith(".pt"):

        path = os.path.join(EMBEDDING_DIR, file)
        vec = torch.load(path)
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy().flatten()
        else:
            vec = vec.flatten()

        # -------------------------------------------------
        # FIX: Correctly extract FULL NAME from file name
        # Example: "divy_tank_1.pt" → "divy tank"
        # -------------------------------------------------
        parts = file.split("_")[:-1]  # remove last item (index)
        student_name = " ".join(parts)

        X.append(vec)
        y.append(student_name.lower())

print("Loaded embeddings:", len(X))

X = np.array(X)
y = np.array(y)

# Normalize embeddings
X = normalize(X, norm="l2")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train k-NN (cosine)
knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
knn.fit(X, y_enc)

# Save model + label encoder
joblib.dump(knn, MODEL_PATH)
joblib.dump(le, LABEL_PATH)

print("🎉 KNN model saved as:", MODEL_PATH)
print("🎉 Label encoder saved as:", LABEL_PATH)
