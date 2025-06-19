from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash
from flask_jwt_extended import create_access_token
import datetime
import os
import joblib

# Directory for artifacts
dir_here = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.abspath(os.path.join(dir_here, '..', 'models'))
user_store_path = os.path.join(MODEL_DIR, 'users.joblib')
users = joblib.load(user_store_path)

# Blueprint for authentication routes
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login endpoint. Expects JSON: {"username": str, "password": str}.
    Returns a JWT access token if credentials are valid.
    """
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    # Basic validation
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400

    # Verify user exists and password matches
    stored_hash = users.get(username)
    if not stored_hash or not check_password_hash(stored_hash, password):
        return jsonify({'error': 'Bad username or password'}), 401

    # Create JWT token (expires in 1 hour)
    expires = datetime.timedelta(hours=1)
    access_token = create_access_token(identity=username, expires_delta=expires)
    return jsonify({'access_token': access_token}), 200
