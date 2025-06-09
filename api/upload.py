# api/upload.py
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

# Build absolute paths to your artifacts directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Load the artifacts once at import time
try:
     lookup_path  = os.path.join(MODEL_DIR, "lookup.joblib")
     mins_path    = os.path.join(MODEL_DIR, "min_prices.joblib")
     model_path   = os.path.join(MODEL_DIR, "model_weights.joblib")  

     lookup                 = joblib.load(lookup_path)
     historical_min_prices  = joblib.load(mins_path)
     best_model             = joblib.load(model_path)
     MIN_TRAIN_PRICE        = min(historical_min_prices.values())
except Exception as e:
     print("Failed to load model artifacts from", MODEL_DIR, ":", e)
     raise

def get_cheapest_price(row):
    mat = row["material_name"]
    if mat not in historical_min_prices:
        raise ValueError(f"No historical price for '{mat}'")
    return float(historical_min_prices[mat]), "historical"

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload-dataset", methods=["POST"])
def upload_dataset():
        if "file" not in request.files:
            return jsonify({"error": "The file is required."}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "File must be a CSV"}), 400

        # save and read
        save_path = os.path.join("/tmp/uploads", file.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        data = pd.read_csv(save_path)

        # 3c) PREPROCESS test‐side data
        text_cols_test = [
            "material_name",
            "material_type",
            "material_subtype",
            "surgeon_name",
            "surgeon_surname",
            "procedure_name",
        ]
        data[text_cols_test] = data[text_cols_test].fillna("missing").astype(str)
        data["surgeon_specific_action"] = data["surgeon_specific_action"].fillna("missing").astype(str)

        data["surgeon_fullname"] = (
            data["surgeon_name"].str.strip() + " " + data["surgeon_surname"].str.strip()
        )
        data["material_name"] = data["material_name"].str.lower()

        # 3d) Build combined_features
        data["combined_features"] = data[
            ["material_name", "material_type", "material_subtype", "surgeon_fullname", "procedure_name"]
        ].agg(" ".join, axis=1)

        # 3e) Create is_default flag
        data["is_default"] = (data["surgeon_name"] == "Standardized").astype(int)

        # 3f) Map material_price from historical_min_prices
        missing_materials = set(data["material_name"]) - set(historical_min_prices.keys())
        if missing_materials:
            return (
                jsonify({"error": f"Materials not found in training data: {missing_materials}"}),
                400,
            )
        #data["material_price"] = data["material_name"].map(historical_min_prices)

        # 3g) Merge with lookup to get train_surgeon_fullname, train_material_price, train_surgeon_specific_action
        data = data.merge(
            lookup,
            left_on=["material_name", "surgeon_fullname"],
            right_on=["material_name", "train_surgeon_fullname"],
            how="left",
        )

        # 3h) PREDICT optimized prices
        # Option A: If your model expects raw features, call preprocess_dataset first:
        # preds_log_series = preprocess_dataset(data)
        # data["predicted_price"] = np.exp(preds_log_series).clip(lower=MIN_TRAIN_PRICE)
        #


        # Option B: If predict_material_prices returns final prices directly from combined_features:
        #data["predicted_price"] = predict_material_prices(data["combined_features"])
        #data["predicted_price"] = data["predicted_price"].clip(lower=MIN_TRAIN_PRICE)
        
        #data["predicted_price"] = best_model.predict(data["combined_features"]).clip(lower=MIN_TRAIN_PRICE)
        
        preds = best_model.predict(data["combined_features"])
        data["predicted_price"] = pd.Series(preds, index=data.index).clip(lower=MIN_TRAIN_PRICE)
        # 3i) RECONSTRUCT per‐procedure results
        results = []
        for proc_id in data["procedure_id"].unique():
            proc_data = data[data["procedure_id"] == proc_id].copy()

            # Identify default materials and surgeon‐added materials
            default_data = proc_data[proc_data["is_default"] == 1]
            default_materials = set(default_data["material_name"])

            surgeon_data = proc_data[proc_data["is_default"] == 0]
            surgeon_added = set(
                surgeon_data[surgeon_data["surgeon_specific_action"] != "default"]["material_name"]
            ) - default_materials

            all_materials = default_materials.union(surgeon_added)

            # Compute original_total_cost = sum(train_material_price for all_materials)
            orig_prices_list = []
            for mat in all_materials:
                mat_rows = proc_data[proc_data["material_name"] == mat]
                if not mat_rows.empty:
                    row0 = mat_rows.iloc[0]
                    orig_prices_list.append(row0["train_material_price"])
            original_total_cost = float(np.sum(orig_prices_list)) if orig_prices_list else 0.0

            # Build optimized_materials dict
            optimized_materials = {}
            for mat in all_materials:
                mat_rows = proc_data[proc_data["material_name"] == mat]
                if mat_rows.empty:
                    continue

                cheapest_price, price_source = get_cheapest_price(mat_rows.iloc[0])
                row0 = mat_rows.iloc[0]
                orig_surgeon_fullname = row0["train_surgeon_fullname"]
                orig_price = row0["train_material_price"]
                orig_action = row0["train_surgeon_specific_action"]

                optimized_materials[mat] = (
                    float(cheapest_price),
                    str(price_source),
                    str(orig_surgeon_fullname),
                    float(orig_price),
                    str(orig_action),
                )

            # Sum optimized_cost across all materials in this procedure
            optimized_cost = float(
                sum(val[0] for val in optimized_materials.values()) if optimized_materials else 0.0
            )

            # Append one entry per material with all required fields (casting to native types)
            for mat, (opt_price, source, orig_surgeon_fullname, orig_price, orig_action) in optimized_materials.items():
                results.append(
                    {
                        "procedure_id":                   int(proc_id),
                        "material_name":                  str(mat),
                        "surgeon_fullname_train":         orig_surgeon_fullname,
                        "original_price_train":           orig_price,
                        "surgeon_specific_action_train":  orig_action,
                        "original_total_cost":            original_total_cost,
                        "optimized_price":                opt_price,
                        "price_source":                   source,
                        "optimized_cost":                 float(round(optimized_cost, 2)),
                    }
                )

        return jsonify({"predictions": results})

   
