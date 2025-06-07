# app.py
from flask import Flask
from api.upload import upload_bp

def create_app():
    app = Flask(__name__)
    # register your blueprint under the root URL (or use a prefix)
    app.debug = True
    app.register_blueprint(upload_bp, url_prefix="")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
