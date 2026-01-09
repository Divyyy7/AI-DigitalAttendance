"""
Seed sample teacher and student for testing.
Run: python create_admin.py
"""
from app import app
from models import db, Teacher, Student
from werkzeug.security import generate_password_hash

with app.app_context():
    db.create_all()
    # sample teacher
    if not Teacher.query.filter_by(name="teacher1").first():
        t = Teacher(name="teacher1", password_hash=generate_password_hash("teach123"), mobile_number="9999999999")
        db.session.add(t)
        db.session.commit()
        print("Created teacher teacher1/teach123")
    else:
        print("Teacher exists")

    # sample student (no images)
    if not Student.query.filter_by(roll_no="s001").first():
        s = Student(full_name="student1", password_hash=generate_password_hash("stud123"), roll_no="s001", mobile_number="9999999999", email="s1@example.com", teacher_id=t.id)
        db.session.add(s)
        db.session.commit()
        print("Created student student1/stud123")
    else:
        print("Student exists")
