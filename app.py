from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os

# Assume these two functions exist and work as expected:
#  - preprocess_dataset(df) → pd.Series (log‐space or raw predictions)
#  - predict_material_prices(series_of_combined_features) → np.ndarray of final prices
from preprocessing import preprocess_dataset
from prediction_model import predict_material_prices

# ================================================================
# 1) LOAD TRAINING DATA AND BUILD LOOKUPS (once, at startup)
# ================================================================
TRAIN_CSV = "data/train/cleaned_data.csv"
TRAIN_DF = pd.read_csv(TRAIN_CSV)

# 1a) Drop invalid train prices
TRAIN_DF = TRAIN_DF.dropna(subset=["material_price"])
TRAIN_DF = TRAIN_DF[TRAIN_DF["material_price"] > 0]

# 1b) Fill missing text columns
text_cols_train = [
    "material_name",
    "material_type",
    "material_subtype",
    "surgeon_name",
    "surgeon_surname",
    "procedure_name",
]
TRAIN_DF[text_cols_train] = TRAIN_DF[text_cols_train].fillna("missing").astype(str)
TRAIN_DF["surgeon_specific_action"] = TRAIN_DF["surgeon_specific_action"].fillna("missing").astype(str)

# 1c) Build “surgeon_fullname” in train set
TRAIN_DF["surgeon_fullname"] = (
    TRAIN_DF["surgeon_name"].str.strip() + " " + TRAIN_DF["surgeon_surname"].str.strip()
)

# 1d) Normalize “material_name” to lowercase
TRAIN_DF["material_name"] = TRAIN_DF["material_name"].str.lower()

# 1e) Build “lookup” DataFrame:
#     (material_name, train_surgeon_fullname) → (train_material_price, train_surgeon_specific_action)
lookup = (
    TRAIN_DF[["material_name", "surgeon_fullname", "material_price", "surgeon_specific_action"]]
    .drop_duplicates(subset=["material_name", "surgeon_fullname"], keep="first")
    .rename(
        columns={
            "surgeon_fullname": "train_surgeon_fullname",
            "material_price": "train_material_price",
            "surgeon_specific_action": "train_surgeon_specific_action",
        }
    )
)

# 1f) Build historical_min_prices: material_name → minimum material_price in train set
historical_min_prices = TRAIN_DF.groupby("material_name")["material_price"].min().to_dict()

# Absolute minimum train‐set price (for clipping)
MIN_TRAIN_PRICE = float(TRAIN_DF["material_price"].min())


# ================================================================
# 2) HELPER: get_cheapest_price(row)
# ================================================================
def get_cheapest_price(row):
    mat = row["material_name"]
    if pd.isna(mat) or mat not in historical_min_prices:
        raise ValueError(f"No historical price found for material '{mat}'")
    return float(historical_min_prices[mat]), "historical"


# ================================================================
# 3) FLASK APP DEFINITION
# ================================================================
app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    try:
        # 3a) Check file presence
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "File must be a CSV"}), 400

        # 3b) Save uploaded CSV
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        data = pd.read_csv(file_path)

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
        data["material_price"] = data["material_name"].map(historical_min_prices)

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
        data["predicted_price"] = predict_material_prices(data["combined_features"])
        data["predicted_price"] = data["predicted_price"].clip(lower=MIN_TRAIN_PRICE)

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

    except Exception as e:
        import traceback
        print("==== ERROR DURING /upload-dataset ====")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
