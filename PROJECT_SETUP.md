# AI-Based Face Recognition Attendance System
Local Setup Guide

--------------------------------------------------
1. SYSTEM REQUIREMENTS
--------------------------------------------------

Operating System:
- Windows 10 / 11 (Recommended)
- Linux (Ubuntu 20.04+)
- macOS (Intel / Apple Silicon)

Hardware Requirements:
- Webcam (mandatory)
- Minimum 8 GB RAM (recommended)
- CPU is sufficient (GPU optional)

--------------------------------------------------
2. SOFTWARE REQUIREMENTS
--------------------------------------------------

Python:
- Python 3.9 or 3.10 (RECOMMENDED)
❌ Do NOT use Python 3.12 (torch issues)

Check version:
python --version

--------------------------------------------------
3. CREATE VIRTUAL ENVIRONMENT
--------------------------------------------------

Windows:
python -m venv env
env\Scripts\activate

Linux / macOS:
python3 -m venv env
source env/bin/activate

--------------------------------------------------
4. INSTALL PYTHON PACKAGES
--------------------------------------------------

pip install --upgrade pip
pip install -r requirements.txt

--------------------------------------------------
5. PROJECT FOLDER STRUCTURE
--------------------------------------------------

cvproject/
│
├── app.py
├── models.py
├── attendance.db
│
├── student_images/
├── embeddings/
│
├── generate_embeddings_per_image.py
├── train_knn.py
├── recognize_knn_attendance.py
│
├── knn_model.joblib
├── label_encoder.joblib
│
├── templates/
│   ├── *.html
│
└── requirements.txt

--------------------------------------------------
6. DATABASE INITIALIZATION
--------------------------------------------------

Run the app once to create database tables:

python app.py

Then stop the server (CTRL + C)

--------------------------------------------------
7. ADD STUDENTS
--------------------------------------------------

- Login as admin
- Add student with multiple images
- Images must contain clear frontal faces
- Embeddings & KNN training will run automatically

--------------------------------------------------
8. START WEBCAM ATTENDANCE
--------------------------------------------------

- Open Webcam page
- Click "Start Stream"
- Face recognition starts
- Attendance marked automatically

--------------------------------------------------
9. VIEW REPORTS
--------------------------------------------------

- Today's Attendance
- Monthly Attendance
- Day-wise Attendance
- Export attendance to Excel

--------------------------------------------------
10. COMMON ERRORS & FIXES
--------------------------------------------------

Torch not installing:
- Use Python 3.9 / 3.10
- Upgrade pip

Webcam not opening:
- Check camera access permissions
- Close other apps using camera

Face not recognized:
- Add more clear images
- Retrain KNN model
- Check distance threshold

--------------------------------------------------
11. OPTIONAL FEATURES
--------------------------------------------------

- Email / SMS integration
- AI agent for attendance analysis
- PDF report generation
- Cloud deployment

--------------------------------------------------
12. RUN APPLICATION
--------------------------------------------------

python app.py

Open browser:
http://127.0.0.1:5000
