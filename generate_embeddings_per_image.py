# generate_embeddings_per_image.py
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = r"C:\Users\HP\OneDrive\cvproject\student_images"
EMBEDDING_DIR = r"C:\Users\HP\OneDrive\cvproject\embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# -----------------------------
# Models
# -----------------------------
mtcnn = MTCNN(keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# -----------------------------
# Process Each Student Folder
# -----------------------------
for student_name in os.listdir(DATASET_PATH):

    student_folder = os.path.join(DATASET_PATH, student_name)
    if not os.path.isdir(student_folder):
        continue

    # ✅ FIX: Use full folder name as student label (Full Name)
    student_label = student_name.strip().lower().replace(" ", "_")

    print(f"\n👩‍🎓 Processing student: {student_name}")

    image_count = 0

    for file in os.listdir(student_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(student_folder, file)

            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"❌ Could not open {file}: {e}")
                continue

            face = mtcnn(img)
            if face is not None:
                emb = resnet(face.unsqueeze(0))

                image_count += 1

                # ✅ Save embedding with student's FULL NAME
                save_path = os.path.join(
                    EMBEDDING_DIR,
                    f"{student_label}_{image_count}.pt"
                )
                torch.save(emb, save_path)
                print(f"✅ Saved {save_path}")

            else:
                print(f"⚠️ No face detected in {file}")

    if image_count == 0:
        print(f"❌ No valid faces found for {student_name}")

print("\n🎯 All students processed successfully!")
