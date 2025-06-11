from app import app
from extensions import db
from models import User  # Make sure your User model is defined in models.py
from werkzeug.security import generate_password_hash

def insert_user(username, raw_password):
    hashed_password = generate_password_hash(raw_password)
    user = User(username=username, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    print(f"User '{username}' added successfully.")

if __name__ == '__main__':
    with app.app_context():
        # CHANGE THESE or read from input/env
        insert_user("admin", "admin123")
