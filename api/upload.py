from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
from extensions import db
from models import Procedure, Material, Surgeon, Speciality, ProcedureMaterial, ProcedureSurgeon
from datetime import datetime, timezone

# Build absolute paths to your artifacts directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Load the artifacts once at import time
try:
    lookup = joblib.load(os.path.join(MODEL_DIR, "lookup.joblib"))
    historical_min_prices = joblib.load(os.path.join(MODEL_DIR, "min_prices.joblib"))
    best_model = joblib.load(os.path.join(MODEL_DIR, "model_weights.joblib"))
    MIN_TRAIN_PRICE = float(min(historical_min_prices.values()))
except Exception as e:
    print("Failed to load model artifacts from", MODEL_DIR, ":", e)
    raise

def get_cheapest_price(material_name: str):
    if material_name not in historical_min_prices:
        raise ValueError(f"No historical price for '{material_name}'")
    return float(historical_min_prices[material_name]), "historical"

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    # 1) File check
    if "file" not in request.files:
        return jsonify({"error": "The file is required."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400

    # 2) Read into DataFrame
    os.makedirs("/tmp/uploads", exist_ok=True)
    path = os.path.join("/tmp/uploads", file.filename)
    file.save(path)
    data = pd.read_csv(path, low_memory=False)

    # 3) Validate required columns
    if "procedure_id" not in data.columns:
        return jsonify({"error": "Missing procedure_id column"}), 400
    if data["material_id"].isnull().any():
        return jsonify({"error": "Missing material_id in input"}), 400

    # 4) Fill text columns
    text_cols = [
        "material_name", "material_type", "material_subtype",
        "surgeon_name", "surgeon_surname", "procedure_name",
    ]
    for col in text_cols + ["surgeon_specific_action"]:
        data[col] = data.get(col, "missing").fillna("missing").astype(str)

    # 5) Build surgeon_fullname, default 'missing missing' → 'unknown'
    data["surgeon_fullname"] = (
        data["surgeon_name"].str.strip() + " " + data["surgeon_surname"].str.strip()
    )
    data.loc[data["surgeon_fullname"] == "missing missing", "surgeon_fullname"] = "unknown"

    # 6) Normalize and combine for model
    data["material_name"] = data["material_name"].str.lower()
    data["combined_features"] = data[
        ["material_name", "material_type", "material_subtype", "surgeon_fullname", "procedure_name"]
    ].agg(" ".join, axis=1)
    data["is_default"] = (data["surgeon_name"] == "Standardized").astype(int)

    # 7) Ensure no unseen materials
    unseen = set(data["material_name"]) - set(historical_min_prices.keys())
    if unseen:
        return jsonify({"error": f"Materials not found in training data: {sorted(unseen)}"}), 400

    # 8) Merge lookup
    data = data.merge(
        lookup,
        left_on=["material_name", "surgeon_fullname"],
        right_on=["material_name", "train_surgeon_fullname"],
        how="left"
    )

    # 9) Fill fallback for missing train_* columns
    data["train_material_price"] = data["train_material_price"].fillna(
        data["material_name"].map(historical_min_prices)
    ).astype(float)
    data["train_surgeon_specific_action"] = data["train_surgeon_specific_action"].fillna("unknown")
    data["train_surgeon_fullname"] = data["train_surgeon_fullname"].fillna(data["surgeon_fullname"])

    # 10) Prediction & clip
    preds = best_model.predict(data["combined_features"])
    data["predicted_price"] = [max(float(p), MIN_TRAIN_PRICE) for p in preds]

    # 11) Compute each procedure’s original cost over unique materials ONLY
    #    Drop duplicate rows for the same procedure & material before summing
    unique_prices = (
        data[['procedure_id', 'material_name', 'train_material_price']]
        .drop_duplicates(subset=['procedure_id', 'material_name'])
    )
    orig_costs = (
        unique_prices
        .groupby('procedure_id')['train_material_price']
        .sum()
        .rename('procedure_original_cost')
    )
    data = data.merge(orig_costs, on='procedure_id', how='left')


    # 12) Build JSON‐serializable results
    results = []
    for proc_id, group in data.groupby("procedure_id"):
        default_set = {
            str(m) for m in group.loc[group["is_default"] == 1, "material_name"]
        }
        added_set = {
            str(m) for m in group.loc[
                (group["is_default"] == 0) & (group["surgeon_specific_action"] != "default"),
                "material_name"
            ]
        } - default_set
        materials = default_set.union(added_set)

        orig_cost = float(group["procedure_original_cost"].iloc[0])
        opt_cost = 0.0

        # first pass: gather each material, accumulate opt_cost
        temp = []
        for mat in materials:
            row = group[group["material_name"] == mat].iloc[0]
            opt_price, source = get_cheapest_price(mat)
            opt_cost += opt_price

            temp.append({
                "material_id":             str(row["material_id"]),
                "material_name":           mat,
                "material_original_price": float(row["train_material_price"]),
                "material_optimized_price": opt_price,
                "material_type":           str(row["material_type"]),
                "material_subtype":        str(row["material_subtype"]),
                "specialty":               str(row.get("speciality") or row.get("specialty") or ""),
                "procedure_id":            int(proc_id),
                "procedure_name":          str(row["procedure_name"]),
                "surgeon_fullname":        str(row["train_surgeon_fullname"]),
                "surgeon_specific_action": str(row["train_surgeon_specific_action"]),
                "price_source":            source,
            })

        # second pass: emit with both costs
        for rec in temp:
            rec["procedure_original_cost"] = orig_cost
            rec["procedure_optimized_cost"] = float(opt_cost)
            results.append(rec)

    # Insert or update records in the database
    for record in results:
        # 1. Speciality (if not exists)
        speciality_name = record.get("specialty")
        speciality = Speciality.query.filter_by(name=speciality_name).first()
        if not speciality:
            speciality = Speciality(name=speciality_name, created_at=datetime.now(timezone.utc))
            db.session.add(speciality)
            db.session.flush()

        # 2. Surgeon (if not exists)
        surgeon = Surgeon.query.filter_by(name=record["surgeon_fullname"]).first()
        if not surgeon:
            surgeon = Surgeon(name=record["surgeon_fullname"], speciality_id=speciality.id, created_at=datetime.now(timezone.utc))
            db.session.add(surgeon)
            db.session.flush()

        # 3. Procedure (if not exists)
        procedure = Procedure.query.filter_by(procedure_id=record["procedure_id"]).first()
        if not procedure:
            procedure = Procedure(
                procedure_id=record["procedure_id"],
                procedure_name=record["procedure_name"],
                original_cost=record["procedure_original_cost"],
                optimized_cost=record["procedure_optimized_cost"],
                speciality_id=speciality.id,
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(procedure)
            db.session.flush()

        # 4. Material (if not exists)
        material = Material.query.filter_by(material_id=record["material_id"]).first()
        if not material:
            material = Material(
                material_id=record["material_id"],
                name=record["material_name"],
                original_price=record["material_original_price"],
                optimized_price=record["material_optimized_price"],
                type=record["material_type"],
                subtype=record["material_subtype"],
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(material)
            db.session.flush()

        # 5. ProcedureMaterial (always insert new)
        proc_mat = ProcedureMaterial(
            procedure_id=procedure.id,
            material_id=material.id,
            surgeon_specific_action=record["surgeon_specific_action"],
            created_at=datetime.now(timezone.utc)
        )
        db.session.add(proc_mat)

        # 6. ProcedureSurgeon (if not exists)
        proc_surgeon = ProcedureSurgeon.query.filter_by(procedure_id=procedure.id, surgeon_id=surgeon.id).first()
        if not proc_surgeon:
            proc_surgeon = ProcedureSurgeon(
                procedure_id=procedure.id,
                surgeon_id=surgeon.id,
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(proc_surgeon)

    # Commit all records
    db.session.commit()

    return jsonify({"predictions": results})
