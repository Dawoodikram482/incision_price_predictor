from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib

# Build absolute paths to your artifacts directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Load the artifacts once at import time
try:
    lookup_path = os.path.join(MODEL_DIR, "lookup.joblib")
    mins_path = os.path.join(MODEL_DIR, "min_prices.joblib")
    model_path = os.path.join(MODEL_DIR, "model_weights.joblib")

    lookup = joblib.load(lookup_path)
    historical_min_prices = joblib.load(mins_path)
    best_model = joblib.load(model_path)
    MIN_TRAIN_PRICE = min(historical_min_prices.values())
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
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400

    # save and read
    save_path = os.path.join("/tmp/uploads", file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    data = pd.read_csv(save_path)

    # Preprocess test-side data
    text_cols_test = [
        "material_id",
        "material_name",
        "material_type",
        "material_subtype",
        "surgeon_name",
        "surgeon_surname",
        "procedure_name",
    ]
    # Fill and enforce str
    data[text_cols_test] = data[text_cols_test].fillna("missing").astype(str)
    data["surgeon_specific_action"] = data["surgeon_specific_action"].fillna("missing").astype(str)

    # Build combined columns
    data["surgeon_fullname"] = (
        data["surgeon_name"].str.strip() + " " + data["surgeon_surname"].str.strip()
    )
    data["material_name"] = data["material_name"].str.lower()

    # Combined features for the model
    data["combined_features"] = data[
        ["material_name", "material_type", "material_subtype", "surgeon_fullname", "procedure_name"]
    ].agg(" ".join, axis=1)

    # Flag for default items
    data["is_default"] = (data["surgeon_name"] == "Standardized").astype(int)

    # Check material_id uniqueness
    if data["material_id"].isnull().any():
        return jsonify({"error": "Missing material_id in input"}), 400

    # Validate materials exist
    missing_materials = set(data["material_name"]) - set(historical_min_prices.keys())
    if missing_materials:
        return (
            jsonify({"error": f"Materials not found in training data: {missing_materials}"}),
            400,
        )

    # Merge with lookup to get train-specific columns
    data = data.merge(
        lookup,
        left_on=["material_name", "surgeon_fullname"],
        right_on=["material_name", "train_surgeon_fullname"],
        how="left",
    )

    # Model prediction and clipping
    preds = best_model.predict(data["combined_features"])
    data["predicted_price"] = pd.Series(preds, index=data.index).clip(lower=MIN_TRAIN_PRICE)

    # Build per-procedure results
    results = []
    for proc_id in data["procedure_id"].unique():
        proc_data = data[data["procedure_id"] == proc_id]

        # Identify defaults vs added
        default_materials = set(proc_data[proc_data["is_default"] == 1]["material_name"])
        surgeon_added = set(
            proc_data[proc_data["is_default"] == 0][proc_data["surgeon_specific_action"] != "default"]["material_name"]
        ) - default_materials
        all_materials = default_materials.union(surgeon_added)

        # Compute original procedure cost
        orig_prices_list = [
            float(proc_data[proc_data["material_name"] == mat].iloc[0]["train_material_price"])
            for mat in all_materials
        ]
        procedure_original_cost = float(np.sum(orig_prices_list)) if orig_prices_list else 0.0

        # Compute optimized prices per material
        optimized_info = {}
        for mat in all_materials:
            row0 = proc_data[proc_data["material_name"] == mat].iloc[0]
            opt_price, source = get_cheapest_price(row0)

            optimized_info[mat] = {
                "material_id": row0["material_id"],
                "material_name": mat,
                "material_original_price": float(row0["train_material_price"]),
                "material_optimized_price": opt_price,
                "material_type": row0["material_type"],
                "material_subtype": row0["material_subtype"],
                "specialty": row0.get("speciality") or row0.get("specialty"),
                "procedure_id": int(proc_id),
                "procedure_name": row0["procedure_name"],
                "surgeon_fullname": row0["train_surgeon_fullname"],
                "surgeon_specific_action": row0["train_surgeon_specific_action"],
                "price_source": source,
            }

        # Sum optimized procedure cost
        procedure_optimized_cost = float(
            sum(info["material_optimized_price"] for info in optimized_info.values())
        )

        # Append each material's record with procedure-level costs
        for mat, info in optimized_info.items():
            record = {
                **info,
                "procedure_original_cost": procedure_original_cost,
                "procedure_optimized_cost": procedure_optimized_cost,
            }
            results.append(record)

    return jsonify({"predictions": results})
