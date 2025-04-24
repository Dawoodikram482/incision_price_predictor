import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('data/cleaned_data.csv', low_memory=False)

# Step 1: Preprocess the Data
data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
data['surgeon_surname'] = data['surgeon_surname'].fillna("Standardized")
data['material_price'] = data['material_price'].fillna(data['material_price'].mean())
data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)

# Encode features
data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])

# Features and target
feature_cols = [col for col in data_encoded.columns if col.startswith('mat_') or col.startswith('action_')] + ['is_default']
X = data_encoded[feature_cols]
y = data_encoded['material_price']

# Initial scaling (for Grid Search and Lasso)
scaler_initial = StandardScaler()
X_scaled = scaler_initial.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fine-tune alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)
best_alpha = grid_search.best_params_['alpha']
print("Best alpha:", best_alpha)

# Train with best alpha (initially on full features)
model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
print("Optimized Ridge Test MSE:", mean_squared_error(y_test, y_pred_test))
print("Optimized Ridge Test R²:", r2_score(y_test, y_pred_test))

# Feature selection with Lasso
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y)
important_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
X_selected = X[important_features]  # Subset to Lasso-selected features before scaling

# Refit scaler on selected features
scaler = StandardScaler()
X_scaled_selected = scaler.fit_transform(X_selected)
X_train_sel, X_test_sel, y_train, y_test = train_test_split(X_scaled_selected, y, test_size=0.2, random_state=42)

# Train model on selected features
model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_train_sel, y_train)
joblib.dump(model, 'model_weights.joblib')
y_pred_test_sel = model.predict(X_test_sel)
print("Test MSE with selected features:", mean_squared_error(y_test, y_pred_test_sel))
baseline_mse = mean_squared_error(y, np.full_like(y, y.mean()))
print("Baseline MSE (Mean Prediction):", baseline_mse)

results = {}
for proc_id in data['procedure_id'].unique():
    proc_data = data[data['procedure_id'] == proc_id]
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

#Output Results for Specific Procedure
specific_proc_id = 1901
print(f"\nProcedure Cost Comparison and Optimization for Procedure ID {specific_proc_id}:")
if specific_proc_id in results:
    info = results[specific_proc_id]
    print(f"Default Cost: ${info['default_cost']:.2f}")
    print(f"Avg Surgeon-Specific Cost: ${info['avg_surgeon_cost']:.2f}")
    print("Optimized Materials:")
    for material, price in info['optimized_materials'].items():
        print(f"Procedure ID: {specific_proc_id} | Material: {material} | Cost: ${price:.2f}")
    print(f"Optimized Cost: ${info['optimized_cost']:.2f}")
else:
    print(f"Procedure ID {specific_proc_id} not found in results.")
# # Save the results to a CSV file
# pd.DataFrame(results).T.to_csv("data/results/ridge_optimization_results.csv")
