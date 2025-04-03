import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data/final_dataset/cleaned_data.csv')

#preprocessing the data
## Fill missing surgeon_name and surgeon_surname with "Standardized"
data['surgeon_name'] = data['surgeon_name'].fillna("Standard")
data['surgeon_surname'] = data['surgeon_surname'].fillna("Standard")

# Debug: Check for NaNs in material_price
print("NaNs in material_price:", data['material_price'].isnull().sum())
print("Sample of material_price:\n", data['material_price'].head())

# Handle NaNs in material_price: adding mean price to rows with NaN
data['material_price'] = data['material_price'].fillna(data['material_price'].mean())
print("Rows after after adding mean prices instead of NaNs:", len(data))

#debug
print("Columns in data: ", data.columns.tolist())
print("Surgeon name values \n ", data['surgeon_name'].value_counts())

# Add is_default column: 1 if Standardized, 0 if surgeon-specific
data['is_default'] = (data['surgeon_name'] == "Standard").astype(int)

#debug
print("Is column created: \n", data[['surgeon_name', 'is_default']].value_counts()) 

#encode material_name and surgeon_specific_action
data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])

#features and target definig
feature_col = [col for col in data_encoded.columns if col.startswith('mat_') or col.startswith('action_')] +['is_default']
X = data_encoded[feature_col]
Y = data_encoded['material_price']

#scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#splitting the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

#fitting data into the model
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)

#evaluating the model
y_pred_train = ridge_model.predict(X_train)
y_pred_test = ridge_model.predict(X_test)

print("Training Mean Squared Error:", mean_squared_error(y_train, y_pred_train))
print("Training R^2 Score:", r2_score(y_train, y_pred_train))
print("Test Mean Squared Error:", mean_squared_error(y_test, y_pred_test))
print("Test R^2 Score:", r2_score(y_test, y_pred_test))

#setting a baseline: predict the mean material price

baseline_mse  = mean_squared_error(Y, np.full_like(Y,Y.mean()))
print("Baseline Mean Squared Error:", baseline_mse)

#comparing prices and optimizing the material lists

results = {} 

for proc_id in data['procedure_id'].unique():
    proc_data = data[data['procedure_id'] == proc_id]
    default_data = proc_data[proc_data['is_default'] == 1]
    default_materials = set(default_data['material_name'])
    default_cost = default_data['material_price'].sum()


    surgeon_data = proc_data[proc_data['is_default'] == 0]
    surgeon_cost = surgeon_data.groupby(['surgeon_name', 'surgeon_surname'])['material_price'].sum()
    avg_surgeon_cost = surgeon_cost.mean() if not surgeon_cost.empty else 0

    proc_encoded = pd.get_dummies(proc_data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
    # proc_X = proc_encoded[feature_col].reindex(columns=feature_col, fill_value=0)
    proc_X = proc_encoded[feature_col].reindex(columns=feature_col, fill_value=0)
    proc_X_scaled = scaler.transform(proc_X)
    predicted_prices = ridge_model.predict(proc_X_scaled)

    surgeon_added = set(surgeon_data[surgeon_data['surgeon_specific_action']!= 'default']['material_name']) - default_materials
    all_materials = default_materials.union(surgeon_added)


    optimized_materials = {}

    for mat in all_materials:
        mat_rows = proc_data[proc_data['material_name'] == mat]
        if not mat_rows.empty:
            mat_idx = proc_data.index[proc_data['material_name'] == mat].tolist()
            pred_price = min(predicted_prices[i] for i in range (len(predicted_prices)) if proc_data.index[i] in mat_idx)
            actual_price = mat_rows['material_price'].min()
            optimized_materials[mat] = min(pred_price, actual_price)
    
    optimized_cost = sum(optimized_materials.values())
    results[proc_id] = {
        'default_cost': default_cost,
        'avg_surgeon_cost': avg_surgeon_cost,
        'optimized_materials': optimized_materials,
        'optimized_cost': optimized_cost,
    }


# Print results
print("\nProcedure Cost Comparison and Optimization:")

for proc_id, info in results.items():
    print(f"Procedure ID: {proc_id}")
    print(f"Default Cost: ${info['default_cost']:.2f}")
    print(f"Avg Surgeon-Specific Cost: ${info['avg_surgeon_cost']:.2f}")
    print(f"Optimized Materials: {info['optimized_materials']}")
    print(f"Optimized Cost: ${info['optimized_cost']:.2f}\n")

# Save the results to a CSV file
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("your_dataset.csv")

# Step 1: Preprocess the Data
# Fill missing surgeon_name and surgeon_surname with "Standardized"
data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
data['surgeon_surname'] = data['surgeon_surname'].fillna("Standardized")

# Debug: Check for NaNs in material_price
print("NaNs in material_price:", data['material_price'].isnull().sum())
print("Sample of material_price:\n", data['material_price'].head())

# Handle NaNs in material_price: Drop rows with NaN
data = data.dropna(subset=['material_price'])
print("Rows after dropping NaNs:", len(data))

# Add is_default column: 1 if Standardized, 0 if surgeon-specific
data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)

# Encode material_name and surgeon_specific_action
data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])

# Features and target
feature_cols = [col for col in data_encoded.columns if col.startswith('mat_') or col.startswith('action_')] + ['is_default']
X = data_encoded[feature_cols]
y = data_encoded['material_price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train Ridge Regression Model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Test MSE:", mean_squared_error(y_test, y_pred_test))

# Baseline
baseline_mse = mean_squared_error(y, np.full_like(y, y.mean()))
print("Baseline MSE (Mean Prediction):", baseline_mse)

# Step 4: Compare Costs and Optimize Material List
results = {}
for proc_id in data['procedure_id'].unique():
    proc_data = data[data['procedure_id'] == proc_id]
    default_data = proc_data[proc_data['is_default'] == 1]
    default_materials = set(default_data['material_name'])
    default_cost = default_data['material_price'].sum()
    
    surgeon_data = proc_data[proc_data['is_default'] == 0]
    surgeon_costs = surgeon_data.groupby(['surgeon_name', 'surgeon_surname'])['material_price'].sum()
    avg_surgeon_cost = surgeon_costs.mean() if not surgeon_costs.empty else 0
    
    proc_encoded = pd.get_dummies(proc_data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
    proc_X = proc_encoded[feature_cols].reindex(columns=feature_cols, fill_value=0)
    proc_X_scaled = scaler.transform(proc_X)
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

# Step 5: Output Results
print("\nProcedure Cost Comparison and Optimization:")
for proc_id, info in results.items():
    print(f"Procedure ID: {proc_id}")
    print(f"Default Cost: ${info['default_cost']:.2f}")
    print(f"Avg Surgeon-Specific Cost: ${info['avg_surgeon_cost']:.2f}")
    print(f"Optimized Materials: {info['optimized_materials']}")
    print(f"Optimized Cost: ${info['optimized_cost']:.2f}\n")

# Save for dashboard
pd.DataFrame(results).T.to_csv("data/results/ridge_optimization_results.csv")