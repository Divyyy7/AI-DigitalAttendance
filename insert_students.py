import os
import torch
from app import app
from models import db, Student, FaceEmbedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_IMAGE_DIR = os.path.join(BASE_DIR, "student_images")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

with app.app_context():

    print("\n🔍 Scanning student folders...\n")

    for folder in os.listdir(STUDENT_IMAGE_DIR):
        folder_path = os.path.join(STUDENT_IMAGE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        roll = folder.strip()
        name = roll.replace("_", " ").title()

        print(f"📁 Student folder found → {roll}")

        # Check student exists
        student = Student.query.filter_by(roll_no=roll).first()
        if not student:
            student = Student(full_name=name, roll_no=roll, image_folder=folder_path)
            db.session.add(student)
            db.session.commit()
            print(f"   ✅ Added student → {name}")
        else:
            print(f"   ⚠️ Student already exists → Skipping insert")

        # Insert embeddings
        count = 0
        for file in os.listdir(EMBED_DIR):
            if file.startswith(roll) and file.endswith(".pt"):

                emb_path = os.path.join(EMBED_DIR, file)

                exists = FaceEmbedding.query.filter_by(
                    student_id=student.id,
                    file_name=file
                ).first()

                if exists:
                    continue

                # Register embedding
                rec = FaceEmbedding(
                    student_id=student.id,
                    file_name=file,
                    embedding_path=emb_path
                )

                db.session.add(rec)
                count += 1

        db.session.commit()
        print(f"   ➕ Saved {count} embeddings\n")

print("🎉 ALL STUDENTS + EMBEDDINGS INSERTED SUCCESSFULLY!\n")
