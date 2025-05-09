import pandas as pd
import numpy as np
from PIL import Image
import joblib

def preprocess_dataset(data, is_training=False):
    """
    Preprocess input dataset for Ridge Regression model.
    Assumes data is a pandas DataFrame with columns: material_name, surgeon_specific_action, surgeon_name.
    """
    # Create is_default column based on surgeon_name (as in ridge_regression.py)
    data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
    data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)

    # One-hot encode categorical columns based on ridge_regression.py
    categorical_cols = ['material_name', 'surgeon_specific_action']
    data_encoded = pd.get_dummies(data, columns=[col for col in categorical_cols if col in data.columns], prefix=['mat', 'action'])
    
    # Load the pipeline components
    pipeline = joblib.load('models/model_weights.joblib')
    feature_names = pipeline['important_features']
    scaler = pipeline['scaler']
    
    # Align the new data with the training feature set
    data_aligned = pd.DataFrame(0, index=data.index, columns=feature_names)
    for col in data_encoded.columns:
        if col in feature_names:
            data_aligned[col] = data_encoded[col]
    data_aligned['is_default'] = data['is_default']
    
    # Apply the scaler
    data_scaled = scaler.transform(data_aligned)
    
    return data_scaled

# def preprocess_image(image_path):
#     """
#     Preprocess image for YOLOv5 inference.
#     Resizes image to 640x640 as expected by YOLOv5.
#     """
#     img = Image.open(image_path)
#     img = img.resize((640, 640))
#     img.save(image_path)
#     return image_path