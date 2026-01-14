from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
import joblib
import numpy as np
from sklearn.preprocessing import normalize

# Load ML models
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

knn = joblib.load("knn_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

UNKNOWN_DISTANCE_THRESHOLD = 0.30
PROBABILITY_THRESHOLD = 1.00

# Helper: Normalize predicted name to match DB

def normalize_name(name):

    name = name.replace("_", " ").strip()
    name = " ".join(name.split())        # remove extra double spaces
    return name.title()                  # capitalize like DB records

# FUNCTION CALLED FROM FLASK /video_feed
def recognize_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    face = mtcnn(img)

    recognized_name = None   # Flask will read this
    debug_text = ""

    if face is not None:

        emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten().reshape(1, -1)
        emb_norm = normalize(emb, norm="l2")

        distances, idxs = knn.kneighbors(emb_norm, n_neighbors=1)
        dist = float(distances[0][0])

        probs = knn.predict_proba(emb_norm)[0]
        best_idx = np.argmax(probs)
        conf = float(probs[best_idx])

        predicted_name_raw = label_encoder.inverse_transform([best_idx])[0]

        # FIX: Normalize so DB can match it correctly
        predicted_name = normalize_name(predicted_name_raw)

        # Unknown logic
        if dist > UNKNOWN_DISTANCE_THRESHOLD or conf < PROBABILITY_THRESHOLD:
            debug_text = f"UNKNOWN (d={dist:.2f}, c={conf:.2f})"
            color = (0, 0, 255)
        else:
            debug_text = f"{predicted_name} (d={dist:.2f}, c={conf:.2f})"
            
            recognized_name = predicted_name_raw 
            color = (0, 255, 0)

        cv2.putText(frame, debug_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    else:
        cv2.putText(frame, "No face detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    return frame, recognized_name