import joblib
import pandas as pd
# import torch
# from fuzzywuzzy import process
# import os

# Load models and dataset
ridge_model = joblib.load("models/model_weights.joblib")
# yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_weights.pt', force_reload=True)
# material_data = pd.read_csv("data/train/cleaned_data.csv")
# material_names = material_data['material_name'].unique().tolist()

def predict_material_prices(data):
    pipeline = joblib.load("models/model_weights.joblib")
    model = pipeline['model']
    predictions = model.predict(data)
    return predictions.tolist()


# def detect_tools(image_path):
#     """
#     Detect surgical tools in an image using YOLOv5.
#     Input: Path to preprocessed image
#     Output: List of detected tool labels and bounding boxes
#     """
#     results = yolov5_model(image_path)
#     detections = results.pandas().xyxy[0]
#     tools = []
#     for _, row in detections.iterrows():
#         tools.append({
#             "label": row['name'],
#             "bbox": [row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]
#         })
#     return tools

# def match_tools(tools):
#     """
#     Match detected tool labels to material names using fuzzy matching.
#     Input: List of detected tools with labels
#     Output: List of matched materials with predicted prices
#     """
#     matched_tools = []
#     for tool in tools:
#         matched_material, score = process.extractOne(tool['label'], material_names)
#         if score > 80:
#             # Create a mock DataFrame for this material to predict its price
#             mock_data = pd.DataFrame([{
#                 "material_name": matched_material,
#                 "specialty": "Ear_Nose_Throat",  # Mock values; adjust based on context
#                 "procedure": "Spinal Injection",
#                 "is_default": 1
#             }])
#             preprocessed_data = preprocess_dataset(mock_data, is_training=False)
#             predicted_price = predict_material_prices(preprocessed_data)[0]
#             matched_tools.append({
#                 "label": tool['label'],
#                 "matched_material": matched_material,
#                 "predicted_price": predicted_price,
#                 "bbox": tool['bbox']
#             })
#     return matched_tools

# def calculate_tray_cost(matched_tools):
#     """
#     Calculate total tray cost from matched tools.
#     Input: List of matched tools with prices
#     Output: Total cost
#     """
#     total_cost = sum(tool['predicted_price'] for tool in matched_tools)
#     return total_cost