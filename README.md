# Incision Price Predictor

A Flask-based web application for predicting procedure prices, managing users, and handling medical data.

## Features

- User authentication with JWT
- Upload and manage procedure and material data
- Predict prices using machine learning models
- RESTful API endpoints for integration

## Project Structure

- `app.py` - Main application entry point
- `models.py` - SQLAlchemy models
- `extensions.py` - Flask extensions (DB, JWT, etc.)
- `api/` - API blueprints (auth, upload, speciality, etc.)
- `scripts/` - Utility scripts (e.g., user store generation)
- `create_db.py` - Script to initialize the database
- `create_user.py` - Script to add a user to the database
- `feature_hasher_transformer.py` - Feature engineering utilities
- `models/` - code for training models and stored weights used for the app
- `Dockerfile`, `docker-compose.yml` - Containerization support

## Setup

1. **Clone the repository**

   ```sh
   git clone <repo-url>
   cd incision_price_predictor
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Set environment variables**

   Create a `.env` file or export variables:

   ```
   DATABASE_URL=postgresql://user:pass@host:port/dbname
   JWT_SECRET_KEY=your-secret-key
   ```

4. **Initialize the database**

   ```sh
   python create_db.py
   ```

5. **Create an admin user**

   ```sh
   python create_user.py
   ```

6. **Run the app**

   ```sh
   python app.py
   ```

   Or with Docker:

   ```sh
   docker-compose up --build
   ```

## API Endpoints

- `/api/auth/*` - Authentication endpoints
- `/api/upload/*` - Data upload endpoints
- `/api/speciality/*` - Speciality data
- `/api/procedure/*` - Procedure data
- `/api/material_breakdown/*` - Material breakdowns
- `/api/procedure_detail/*` - Procedure details
- `/api/procedure_summary/*` - Procedure summaries

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

For more details, see the source files and API blueprints in the [api/](api/) directory.