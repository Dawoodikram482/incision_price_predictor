from flask import Flask, request, jsonify
import pandas as pd
import os
from preprocessing import preprocess_dataset, preprocess_image
from prediction_model import predict_material_prices, detect_tools, match_tools, calculate_tray_cost

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Load and preprocess dataset (assume testing data, no prices)
    data = pd.read_csv(file_path)
    preprocessed_data = preprocess_dataset(data, is_training=False)
    
    # Predict prices
    predictions = predict_material_prices(preprocessed_data)
    
    # Group predictions by procedure (mock procedure IDs for simplicity)
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            "procedure_id": i + 1901,  # Mock ID
            "material_name": data['material_name'].iloc[i],
            "specialty": data['specialty'].iloc[i],
            "procedure": data['procedure'].iloc[i],
            "is_default": data['is_default'].iloc[i],
            "predicted_price": pred
        })
    
    return jsonify({"predictions": results})

# @app.route('/tray-cost', methods=['POST'])
# def tray_cost():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400
    
#     image = request.files['image']
#     image_path = os.path.join(UPLOAD_FOLDER, image.filename)
#     image.save(image_path)
    
#     # Preprocess image
#     preprocess_image(image_path)
    
#     # Detect tools
#     tools = detect_tools(image_path)
    
#     # Match tools to materials and predict prices
#     matched_tools = match_tools(tools)
    
#     # Calculate total cost
#     total_cost = calculate_tray_cost(matched_tools)
    
#     return jsonify({
#         "detected_tools": matched_tools,
#         "total_cost": total_cost,
#         "image_path": image_path
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)