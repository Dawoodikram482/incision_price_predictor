from app import app
from extensions import db
from models import Speciality, Surgeon, Procedure, Material, ProcedureMaterial, ProcedureSurgeon

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created!")
