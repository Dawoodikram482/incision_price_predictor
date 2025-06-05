from flask import Flask, request, jsonify
import pandas as pd
import os
from preprocessing import preprocess_dataset
from prediction_model import predict_material_prices


app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path)

        # Preprocessing and model prediction
        prediction_series = preprocess_dataset(data)
        data['optimized_cost'] = prediction_series

        # Optionally include original price if available
        has_prices = 'price' in data.columns
        results = []
        for i, row in data.iterrows():
            result = {
                "procedure_id":               row.get("procedure_id"),
                "material_name":              row.get("material_name"),
                "surgeon_fullname_train":     row.get("surgeon_fullname_train"),
                "original_price_train":       row.get("original_price_train"),
                "surgeon_specific_action":    row.get("surgeon_specific_action_train"),
                "original_total_cost":        row.get("original_total_cost"),
                "optimized_price":            row.get("optimized_price"),
                "optimized_cost":             round(row['optimized_cost'], 2),
            }

            if has_prices:
                result["default_cost"] = row['price']
            results.append(result)

        return jsonify({"predictions": results})

    except Exception as e:
        import traceback
        print("==== ERROR DURING /upload-dataset ====")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

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
    
#     # Match tools to materials
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