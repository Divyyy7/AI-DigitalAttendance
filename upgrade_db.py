import sqlite3

conn = sqlite3.connect("attendance.db")
cur = conn.cursor()

try:
    cur.execute("ALTER TABLE student ADD COLUMN mobile TEXT;")
except:
    print("Column 'mobile' already exists.")

try:
    cur.execute("ALTER TABLE student ADD COLUMN email TEXT;")
except:
    print("Column 'email' already exists.")

conn.commit()
conn.close()

print("Database upgraded successfully.")
