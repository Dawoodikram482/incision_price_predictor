from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash
from flask_jwt_extended import create_access_token
import datetime

# === Static user credentials ===
# This is the same hashed password you inserted into the DB manually
STORED_USERS = {
    "admin": "scrypt:32768:8:1$V6V3a27xPhk4PF9v$10828fa7c3e4fad9dfbd046f1a880515fe43fae6aa8295bb1ae314fed32e234e09ef8154e60cd3e3f4358f7ab1bcf6ef31205bf69adf54ece8d92ce18a9b7c27"  # Replace with your actual hash
}

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

    if not username or not password:
        return jsonify({'error': 'Please enter username and password'}), 400

    stored_hash = STORED_USERS.get(username)
    if not stored_hash or not check_password_hash(stored_hash, password):
        return jsonify({'error': 'Invalid username or password'}), 401

    # Create JWT token (expires in 1 hour)
    expires = datetime.timedelta(hours=1)
    access_token = create_access_token(identity=username, expires_delta=expires)
    return jsonify({'access_token': access_token}), 200
