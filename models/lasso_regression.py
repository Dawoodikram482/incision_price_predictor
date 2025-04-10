import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

print("Starting script execution...")

# Load and preprocess data
print("Loading data from 'data/final_dataset/cleaned_data.csv'...")
data = pd.read_csv('data/final_dataset/cleaned_data.csv', low_memory=False)
print("Data loaded. Shape:", data.shape)

data = data.sample(frac=0.2, random_state=42) # Sample 20% of the data for faster processing
print("Data sampled. New shape:", data.shape)

data['surgeon_name'] = data['surgeon_name'].fillna("Standardized")
data['surgeon_surname'] = data['surgeon_surname'].fillna("Standardized")
data['material_price'] = data['material_price'].fillna(data['material_price'].mean())
data['is_default'] = (data['surgeon_name'] == "Standardized").astype(int)
print("Preprocessing complete. Starting encoding...")
data_encoded = pd.get_dummies(data, columns=['material_name', 'surgeon_specific_action'], prefix=['mat', 'action'])
print("Encoding complete. Shape:", data_encoded.shape)

# Features and target
feature_cols = [col for col in data_encoded.columns if col.startswith('mat_') or col.startswith('action_')] + ['is_default']
X = data_encoded[feature_cols]
y = data_encoded['material_price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Fine-tune alpha
param_grid = {'alpha': [5.0, 7.5, 10.0, 12.5, 15.0]}
grid_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
print("Starting Grid Search for Ridge alpha...")
grid_search.fit(X_scaled, y)
best_alpha = grid_search.best_params_['alpha']
model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
print("Optimized Ridge Test MSE:", mean_squared_error(y_test, y_pred_test))
print("Optimized Ridge Test R²:", r2_score(y_test, y_pred_test))

# Tune Lasso
param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1]}
lasso_search = GridSearchCV(Lasso(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
print("Starting Grid Search for Lasso alpha...")
lasso_search.fit(X_scaled, y)
important_features = [feature_cols[i] for i in range(len(feature_cols)) if lasso_search.best_estimator_.coef_[i] != 0]
print("Number of selected features:", len(important_features))
X_scaled_selected = X_scaled[:, [feature_cols.index(f) for f in important_features]]
X_train_sel, X_test_sel = X_train[:, [feature_cols.index(f) for f in important_features]], X_test[:, [feature_cols.index(f) for f in important_features]]
model.fit(X_train_sel, y_train)
y_pred_test_sel = model.predict(X_test_sel)
print("Test MSE with selected features:", mean_squared_error(y_test, y_pred_test_sel))

print("Script execution complete!")