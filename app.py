# app.py
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from api.upload import upload_bp
from api.speciality import speciality_bp
from api.procedure_detail import procedure_detail_bp
from api.procedure_summary import procedure_summary_bp
from api.material_breakdown import material_breakdown_bp
from api.auth import auth_bp
from api.surgeon import surgeon_bp
from extensions import db
import os
import datetime


def create_app():
    app = Flask(__name__)

    # CORS configuration
    CORS(app)

    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # JWT configuration
    # Use a strong secret key in production; set via environment variable
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'change-this-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)

    db.init_app(app)

    # Initialize JWT manager
    JWTManager(app)

    # Register authentication and API blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(speciality_bp, url_prefix='/api')
    app.register_blueprint(procedure_detail_bp, url_prefix='/api')
    app.register_blueprint(procedure_summary_bp, url_prefix='/api')
    app.register_blueprint(material_breakdown_bp, url_prefix='/api')
    app.register_blueprint(surgeon_bp, url_prefix="/api")


    app.debug = True
    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)