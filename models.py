from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(120), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    mobile = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    image_folder = db.Column(db.String(300), nullable=False)  # folder storing multiple face images

    # relationships
    embeddings = db.relationship('FaceEmbedding', backref='student', cascade="all, delete-orphan")
    attendance = db.relationship('Attendance', backref='student', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student {self.roll_no}>"


class FaceEmbedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

    file_name = db.Column(db.String(200), nullable=False)
    embedding_path = db.Column(db.String(300), nullable=False)

    def __repr__(self):
        return f"<Embedding {self.file_name}>"


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

    date = db.Column(db.Date, nullable=False)
    day_of_week = db.Column(db.String(20))
    status = db.Column(db.String(20))     # Present / Absent
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(50))     # webcam / manual

    def __repr__(self):
        return f"<Attendance {self.student_id} {self.date}>"
