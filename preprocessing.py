import pandas as pd
import numpy as np
from PIL import Image
import joblib

def preprocess_dataset(data):
    """
    Preprocess the data and return a Series of actual price predictions.
    The model outputs log-prices, so we exponentiate to get back to real prices.
    """
    text_columns = [
        'material_name',
        'material_type',
        'material_subtype',
        'surgeon_name',
        'procedure_name'
    ]

    # Clean text columns
    data[text_columns] = data[text_columns].fillna('missing').astype(str)

    # Create combined text feature
    data['combined_features'] = data[text_columns].agg(' '.join, axis=1)

    # Load trained pipeline
    bundle = joblib.load('models/model_weights.joblib')
    pipeline = bundle['model'] if isinstance(bundle, dict) else bundle

    # Predict log(price), then back-transform
    log_preds = pipeline.predict(data['combined_features'])
    price_preds = np.exp(log_preds)

    # Return Series aligned with input data
    return pd.Series(price_preds, index=data.index, name='prediction')

# def preprocess_image(image_path):
#     """
#     Preprocess image for YOLOv5 inference.
#     Resizes image to 640x640 as expected by YOLOv5.
#     """
#     img = Image.open(image_path)
#     img = img.resize((640, 640))
#     img.save(image_path)
#     return image_path