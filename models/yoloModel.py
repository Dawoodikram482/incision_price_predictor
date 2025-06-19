from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from transformers import BertTokenizer, BertModel
import os
from sklearn.metrics.pairwise import cosine_similarity
import time

# Start timing
start_time = time.time()

# Load price dataset
print("Loading price dataset...")
data = pd.read_csv('data/cvData/materials.csv', low_memory=False)
price_map = dict(zip(data['material_name'], data['material_price']))

# Load obj.names
print("Loading obj.names...")
with open('data/cvData/obj.names', 'r') as f:
    yolo_class_names = [line.strip() for line in f]

# NLP setup for text embeddings
print("Setting up BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_nlp = BertModel.from_pretrained('bert-base-multilingual-cased')
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_nlp(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Generate embeddings for YOLO class names
print("Generating embeddings for YOLO class names...")
yolo_embeddings = np.array([get_text_embedding(name) for name in yolo_class_names])

# Generate embeddings for cleaned_data.csv material names
print("Generating embeddings for material names...")
material_names = data['material_name'].tolist()
material_embeddings = np.array([get_text_embedding(name) for name in material_names])

# Match YOLO names to material names using cosine similarity
def match_names(yolo_name, yolo_emb, material_embs, material_names):
    yolo_emb = yolo_emb.reshape(1, -1)
    similarities = cosine_similarity(yolo_emb, material_embs)
    best_match_idx = np.argmax(similarities)
    return material_names[best_match_idx], similarities[0][best_match_idx]

# Load trained YOLOv8 model
print("Loading YOLO model...")
yolo_model = YOLO('runs/detect/train20/weights/best.pt')  # Adjust based on your training run

# Inference pipeline
def predict_tray_cost(image_path: str, yolo_model, price_map, yolo_class_names, yolo_embeddings, material_embeddings, material_names) -> dict:
    print(f"Predicting on {image_path}...")
    results = yolo_model.predict(image_path, imgsz=640)
    total_cost = 0
    detections = []
    
    if results and len(results) > 0 and hasattr(results[0], 'boxes'):
        print(f"Found {len(results[0].boxes)} detections...")
        for box in results[0].boxes:
            cls = int(box.cls.item())
            confidence = box.conf.item()
            yolo_name = yolo_class_names[cls]
            matched_name, similarity = match_names(yolo_name, yolo_embeddings[cls], material_embeddings, material_names)
            price = price_map.get(matched_name, data['material_price'].median())
            total_cost += price
            detections.append({
                "instrument": yolo_name,
                "matched_material": matched_name,
                "confidence": confidence,
                "price": f"${price:.2f}",
                "similarity": f"{similarity:.3f}"
            })
    else:
        print("No detections found.")
    
    return {"total_cost": f"${total_cost:.2f}", "detections": detections}

# Example usage
try:
    print("Starting prediction...")
    result = predict_tray_cost('data/cvData/obj_train_data/images/tray1.png', yolo_model, price_map, yolo_class_names, yolo_embeddings, material_embeddings, material_names)
    print("Result:", result)
except Exception as e:
    print(f"Error occurred: {e}")

# End timing
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")