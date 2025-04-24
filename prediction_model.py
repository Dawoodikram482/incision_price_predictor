import joblib
import numpy as np

def load_model():
    # Load the trained model and scaler
    model = joblib.load('model_weights.joblib')
    scaler = joblib.load('scaler.joblib')
    # Load the feature columns used during training
    feature_cols = joblib.load('feature_cols.joblib')
    return model, scaler, feature_cols

def predict_price(model, processed_data):
    # Make prediction using the loaded model
    return model.predict(processed_data)