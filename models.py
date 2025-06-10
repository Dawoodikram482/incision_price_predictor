from extensions import db
from datetime import datetime

class Speciality(db.Model):
    __tablename__ = 'specialities'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f'<Speciality {self.name}>'


class Surgeon(db.Model):
    __tablename__ = 'surgeons'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    speciality_id = db.Column(db.Integer, db.ForeignKey('specialities.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    speciality = db.relationship('Speciality', backref=db.backref('surgeons', lazy=True))

    def __repr__(self):
        return f'<Surgeon {self.name}>'


class Procedure(db.Model):
    __tablename__ = 'procedures'

    id = db.Column(db.Integer, primary_key=True)
    procedure_id = db.Column(db.BigInteger, nullable=False)
    procedure_name = db.Column(db.String(200), nullable=False)
    original_cost = db.Column(db.Float, nullable=False)
    optimized_cost = db.Column(db.Float, nullable=False)
    speciality_id = db.Column(db.Integer, db.ForeignKey('specialities.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    speciality = db.relationship('Speciality', backref=db.backref('procedures', lazy=True))
    
    def __repr__(self):
        return f'<Procedure {self.procedure_name}>'


class Material(db.Model):
    __tablename__ = 'materials'

    id = db.Column(db.Integer, primary_key=True)
    material_id = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    original_price = db.Column(db.Float, nullable=False)
    optimized_price = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(100), nullable=False)
    subtype = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f'<Material {self.name}>'


class ProcedureMaterial(db.Model):
    __tablename__ = 'procedure_materials'

    id = db.Column(db.Integer, primary_key=True)
    procedure_id = db.Column(db.Integer, db.ForeignKey('procedures.id'), nullable=False)
    material_id = db.Column(db.Integer, db.ForeignKey('materials.id'), nullable=False)
    surgeon_specific_action = db.Column(db.Enum('DEFAULT', 'ADDED', name='surgeon_action'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    procedure = db.relationship('Procedure', backref=db.backref('procedure_materials', lazy=True))
    material = db.relationship('Material', backref=db.backref('procedure_materials', lazy=True))

    def __repr__(self):
        return f'<ProcedureMaterial Procedure={self.procedure_id} Material={self.material_id}>'


class ProcedureSurgeon(db.Model):
    __tablename__ = 'procedure_surgeons'

    id = db.Column(db.Integer, primary_key=True)
    procedure_id = db.Column(db.Integer, db.ForeignKey('procedures.id'), nullable=False)
    surgeon_id = db.Column(db.Integer, db.ForeignKey('surgeons.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    procedure = db.relationship('Procedure', backref=db.backref('procedure_surgeons', lazy=True))
    surgeon = db.relationship('Surgeon', backref=db.backref('procedure_surgeons', lazy=True))

    def __repr__(self):
        return f'<ProcedureSurgeon Procedure={self.procedure_id} Surgeon={self.surgeon_id}>'
