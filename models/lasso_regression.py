# import pandas as pd
# import numpy as np
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score

# print("Starting script execution...")

# # Load and preprocess data
# print("Loading data from 'data/final_dataset/cleaned_data.csv'...")
# data = pd.read_csv('data/final_dataset/cleaned_data.csv', low_memory=False)
# print("Data loaded. Shape:", data.shape)

# data = data.sample(frac=0.2, random_state=42) # Sample 20% of the data for faster processing
# print("Data sampled. New shape:", data.shape)

# data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
# data['surgeon_surname'] = data['surgeon_surname'].fillna("Standardized")
# data['material_price'] = data['material_price'].fillna(data['material_price'].mean())
# data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)
# print("Preprocessing complete. Starting encoding...")
# data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
# print("Encoding complete. Shape:", data_encoded.shape)

# # Features and target
# feature_cols = [col for col in data_encoded.columns if col.startswith('mat_') or col.startswith('action_')] + ['is_default']
# X = data_encoded[feature_cols]
# y = data_encoded['material_price']

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# print("Data split complete.")

# # Fine-tune alpha
# param_grid = {'alpha': [5.0, 7.5, 10.0, 12.5, 15.0]}
# grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
# print("Starting Grid Search for Ridge alpha...")
# grid_search.fit(X_scaled, y)
# best_alpha = grid_search.best_params_['alpha']
# model = Ridge(alpha=best_alpha, random_state=42)
# model.fit(X_train, y_train)
# y_pred_test = model.predict(X_test)
# print("Optimized Ridge Test MSE:", mean_squared_error(y_test, y_pred_test))
# print("Optimized Ridge Test R²:", r2_score(y_test, y_pred_test))

# # Tune Lasso
# param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1]}
# lasso_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
# print("Starting Grid Search for Lasso alpha...")
# lasso_search.fit(X_scaled, y)
# important_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso_search.best_estimator_.coef_[i] != 0]
# print("Number of selected features:", len(important_features))
# X_scaled_selected = X_scaled[:, [feature_cols.index(f) for f in important_features]]
# X_train_sel, X_test_sel = X_train[:, [feature_cols.index(f) for f in important_features]], X_test[:, [feature_cols.index(f) for f in important_features]]
# model.fit(X_train_sel, y_train)
# y_pred_test_sel = model.predict(X_test_sel)
# print("Test MSE with selected features:", mean_squared_error(y_test, y_pred_test_sel))

# print("Script execution complete!")

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Function to calculate Adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Load datasets
train_data = pd.read_csv('data/train/cleaned_data.csv', low_memory=False)
test_data = pd.read_csv('data/test/new_data.csv', low_memory=False)

# Step 1: Preprocess Training Data
train_data['surgeon_name'] = train_data['surgeon_name'].fillna("Standardized")
train_data['surgeon_surname'] = train_data['surgeon_surname'].fillna("Standardized")
train_data['is_default'] = (train_data['surgeon_name'] == "Standardized").astype(int)

# Preprocess Test Data (no material_price)
test_data['surgeon_name'] = test_data['surgeon_name'].fillna("Standardized")
test_data['surgeon_surname'] = test_data['surgeon_surname'].fillna("Standardized")
test_data['is_default'] = (test_data['surgeon_name'] == "Standardized").astype(int)

# Encode features for both datasets
train_encoded = pd.get_dummies(train_data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
test_encoded = pd.get_dummies(test_data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])

# Align test features with training features
feature_cols = [col for col in train_encoded.columns if col.startswith('mat_') or col.startswith('action_')] + ['is_default']
test_encoded = test_encoded.reindex(columns=feature_cols, fill_value=0)

# Features and target for training
X_train_full = train_encoded[feature_cols]
y_train = train_encoded['material_price']

# Features for testing
X_test = test_encoded[feature_cols]

# Initial scaling for Grid Search and Lasso
scaler_initial = StandardScaler()
X_train_scaled = scaler_initial.fit_transform(X_train_full)
X_test_scaled = scaler_initial.transform(X_test)

# Step 2: Train-Test Split for Validation (from training data)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Step 3: Fine-tune Ridge alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
# print("Best Ridge alpha:", best_alpha)

# Step 4: Feature selection with Lasso
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train_scaled, y_train)
important_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
# print(f"Selected {len(important_features)} features with Lasso")

# Subset to selected features
X_train_selected = X_train_full[important_features]
X_test_selected = X_test[important_features]

# Refit scaler on selected features
scaler = StandardScaler()
X_train_scaled_selected = scaler.fit_transform(X_train_selected)
X_test_scaled_selected = scaler.transform(X_test_selected)

# Train-Test Split for selected features
X_train_split_sel, X_val_split_sel, y_train_split, y_val_split = train_test_split(X_train_scaled_selected, y_train, test_size=0.2, random_state=42)

# Step 5: Train Ridge model on selected features
model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_train_split_sel, y_train_split)
# joblib.dump(model, 'model_weights.joblib')

# Step 6: Evaluate model on training and validation sets
y_train_pred = model.predict(X_train_split_sel)
y_val_pred = model.predict(X_val_split_sel)

# Training metrics
train_mse = mean_squared_error(y_train_split, y_train_pred)
train_r2 = r2_score(y_train_split, y_train_pred)
train_adj_r2 = adjusted_r2(train_r2, X_train_split_sel.shape[0], X_train_split_sel.shape[1])

# Validation metrics
val_mse = mean_squared_error(y_val_split, y_val_pred)
val_r2 = r2_score(y_val_split, y_val_pred)
val_adj_r2 = adjusted_r2(val_r2, X_val_split_sel.shape[0], X_val_split_sel.shape[1])

print("\nTraining Set Metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"R²: {train_r2:.4f}")
print(f"Adjusted R²: {train_adj_r2:.4f}")

print("\nValidation Set Metrics:")
print(f"MSE: {val_mse:.2f}")
print(f"R²: {val_r2:.4f}")
print(f"Adjusted R²: {val_adj_r2:.4f}")

# Step 7: Predict prices for test dataset
test_encoded['predicted_price'] = model.predict(X_test_scaled_selected)

# Step 8: Optimize costs for test dataset
results = {}
for proc_id in test_data['procedure_id'].unique():
    proc_data = test_data[test_data['procedure_id'] == proc_id]
    default_data = proc_data[proc_data['is_default'] == 1]
    default_materials = set(default_data['material_name'])
    default_cost = default_data['material_price'].sum()
    
    surgeon_data = proc_data[proc_data['is_default'] == 0]
    surgeon_costs = surgeon_data.groupby(['surgeon_name', 'surgeon_surname'])['material_price'].sum()
    avg_surgeon_cost = surgeon_costs.mean() if not surgeon_costs.empty else 0
    
    # Encode procedure data and align with important_features
    proc_encoded = pd.get_dummies(proc_data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
    proc_X = proc_encoded.reindex(columns=important_features, fill_value=0)
    proc_X_scaled = scaler.transform(proc_X)  # Use the scaler fitted on important_features
    predicted_prices = model.predict(proc_X_scaled)
    
    surgeon_added = set(surgeon_data[surgeon_data['surgeon_specific_action'] != 'default']['material_name']) - default_materials
    all_materials = default_materials.union(surgeon_added)
    
    optimized_materials = {}
    for mat in all_materials:
        mat_rows = proc_data[proc_data['material_name'] == mat]
        if not mat_rows.empty:
            mat_idx = proc_data.index[proc_data['material_name'] == mat].tolist()
            pred_price = min(predicted_prices[i] for i in range(len(predicted_prices)) if proc_data.index[i] in mat_idx)
            actual_price = mat_rows['material_price'].min()
            optimized_materials[mat] = min(pred_price, actual_price)
    
    optimized_cost = sum(optimized_materials.values())
    results[proc_id] = {
        'default_cost': default_cost,
        'avg_surgeon_cost': avg_surgeon_cost,
        'optimized_materials': optimized_materials,
        'optimized_cost': optimized_cost
    }
# Step 9: Output results for specific procedure
# specific_proc_id = 1901
# print(f"\nProcedure Cost Optimization for Procedure ID {specific_proc_id}:")
# if specific_proc_id in results:
#     info = results[specific_proc_id]
#     print("Optimized Materials:")
#     for material, price in info['optimized_materials'].items():
#         if price != float('inf'):
#             print(f"Procedure ID: {specific_proc_id} | Material: {material} | Predicted Cost: ${price:.2f}")
#     print(f"Optimized Cost: ${info['optimized_cost']:.2f}")
# else:
#     print(f"Procedure ID {specific_proc_id} not found in results.")

# Save results to CSV
results_df = pd.DataFrame([
    {
        'procedure_id': proc_id,
        'material': mat,
        'predicted_cost': price,
        'optimized_cost': info['optimized_cost']
    }
    for proc_id, info in results.items()
    for mat, price in info['optimized_materials'].items()
    if price != float('inf')
])
results_df.to_csv("data/results/test_optimization_results.csv", index=False)