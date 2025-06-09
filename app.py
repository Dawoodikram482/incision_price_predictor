# app.py
from flask import Flask
from flask_cors import CORS
from api.upload import upload_bp
from extensions import db
import os

def create_app():
    app = Flask(__name__)

    # cors configuration
    CORS(app)

    # Databse configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # register your blueprint under the root URL (or use a prefix)
    app.debug = True
    app.register_blueprint(upload_bp, url_prefix="")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
