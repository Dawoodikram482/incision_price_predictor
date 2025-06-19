import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/final_dataset/cleaned_data.csv')

# Pivot to procedure x material matrix
pivot_data = pd.pivot_table(data, 
                            index='procedure_id', 
                            columns='material_name', 
                            values='material_price', 
                            aggfunc='mean', 
                            fill_value=0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_data)

# Train K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
pivot_data['cluster'] = labels

# Evaluate
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Baseline
baseline_inertia = ((X_scaled - X_scaled.mean(axis=0)) ** 2).sum()
print("Baseline Inertia:", baseline_inertia)

# Elbow method
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show(block=False)

# Analyze clusters
cluster_summary = pivot_data.groupby('cluster').mean()
print(cluster_summary)

# Cost per cluster
original_with_clusters = data.merge(pivot_data['cluster'], left_on='procedure_id', right_index=True)
print(original_with_clusters.groupby('cluster')['total_procedure_price'].mean())