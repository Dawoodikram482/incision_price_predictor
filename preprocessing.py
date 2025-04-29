import pandas as pd
import numpy as np
from PIL import Image

def preprocess_dataset(data, is_training=True):
    """
    Preprocess input dataset for Ridge Regression model.
    Args:
        data (pd.DataFrame): Input dataset
        is_training (bool): If True, assumes price column exists (training); if False, no price column (testing)
    Returns:
        pd.DataFrame: Preprocessed dataset ready for prediction
    """
    # One-hot encode categorical columns
    categorical_cols = ['material_name', 'specialty', 'procedure', 'is_default']
    data_encoded = pd.get_dummies(data, columns=[col for col in categorical_cols if col in data.columns])
    
    # Load training data to get expected columns
    train_data = pd.read_csv('data/train/cleaned_data.csv')
    train_encoded = pd.get_dummies(train_data, columns=categorical_cols)
    
    # Ensure test data has the same columns as training data (except price)
    expected_columns = [col for col in train_encoded.columns if col != 'price']
    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    
    # Reorder columns to match training data
    data_encoded = data_encoded[expected_columns]
    
    return data_encoded

# def preprocess_image(image_path):
#     """
#     Preprocess image for YOLOv5 inference.
#     Resizes image to 640x640 as expected by YOLOv5.
#     """
#     img = Image.open(image_path)
#     img = img.resize((640, 640))
#     img.save(image_path)
#     return image_path