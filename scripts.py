import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_dataset

# Load the pipeline components
pipeline = joblib.load("models/model_weights.joblib")
model = pipeline['model']

# Load the new dataset
new_data_path = "data/test/new_data.csv"  # Provided by the client
new_data = pd.read_csv(new_data_path, low_memory=False)

# Preprocess the new dataset
preprocessed_data = preprocess_dataset(new_data)

# Make predictions
predictions = model.predict(preprocessed_data)

# Display results
# Reconstruct results with original data for readability
results = pd.DataFrame({
    "material_name": new_data["material_name"],
    "surgeon_specific_action": new_data.get("surgeon_specific_action", ["default"] * len(new_data)),
    "surgeon_name": new_data.get("surgeon_name", ["Standardized"] * len(new_data)),
    "is_default": new_data["is_default"],
    "predicted_price": predictions
})

# If the dataset has actual prices, compare them
if "material_price" in new_data.columns:
    results["actual_price"] = new_data["material_price"]
    results["error"] = abs(results["actual_price"] - results["predicted_price"])

print("Prediction Results:")
print(results)

# Optional: Save results to a CSV
results.to_csv("prediction_results.csv", index=False)