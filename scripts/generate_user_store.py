# scripts/generate_user_store.py
"""
Connects to your SQL database, reads the users table, hashes passwords (if not already hashed),
and writes a joblib file at models/users.joblib containing a dict {username: password_hash}.

Usage:
    export DATABASE_URL="postgresql://user:pass@host:port/dbname"
    python scripts/generate_user_store.py
"""
import os
import joblib
from sqlalchemy import create_engine, MetaData, Table, select
from werkzeug.security import generate_password_hash

# Load database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise RuntimeError('DATABASE_URL environment variable not set')

# Paths
here = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.abspath(os.path.join(here, '..', 'models'))
OUTPUT_PATH = os.path.join(MODEL_DIR, 'users.joblib')

# Connect to database
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Reflect the users table (columns: 'username', 'password')
users_table = Table(
    'users', metadata,
    autoload_with=engine
)

# Query all users
with engine.connect() as conn:
    stmt = select(
        users_table.c.username,
        users_table.c.password
    )
    results = conn.execute(stmt).fetchall()

# Build user store: hash passwords if needed
user_store = {}
for username, pwd in results:
    # If pwd is already a hash, preserve it
    if isinstance(pwd, str) and (pwd.startswith('pbkdf2:') or pwd.startswith('sha256:')):
        pwd_hash = pwd
    else:
        pwd_hash = generate_password_hash(pwd)
    user_store[username] = pwd_hash

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
# Save to joblib
joblib.dump(user_store, OUTPUT_PATH)
print(f"Wrote user store to {OUTPUT_PATH}")
