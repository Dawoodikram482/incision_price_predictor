import pandas as pd
import numpy as np
from PIL import Image

def preprocess_input(input_data, scaler, feature_cols):
    # Convert input JSON to DataFrame
    data = pd.DataFrame([input_data])

    # Step 1: Handle missing values
    data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
    data['surgeon_surname'] = data['surgeon_surname'].fillna("Standardized")
    data['material_price'] = data['material_price'].fillna(data['material_price'].mean()) if 'material_price' in data else 0
    data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)

    # Step 2: Encode categorical features
    data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])

    # Step 3: Ensure all expected feature columns are present
    for col in feature_cols:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    # Keep only the expected feature columns in the correct order
    data_encoded = data_encoded[feature_cols]

    # Step 4: Scale the features
    data_scaled = scaler.transform(data_encoded)

    return data_scaled

# def preprocess_image(image_path):
#     """
#     Preprocess image for YOLOv5 inference.
#     Resizes image to 640x640 as expected by YOLOv5.
#     """
#     img = Image.open(image_path)
#     img = img.resize((640, 640))  # YOLOv5 expects 640x640 input
#     img.save(image_path)  # Overwrite with resized image
#     return image_path