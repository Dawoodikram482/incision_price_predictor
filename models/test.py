from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import os
import joblib

# Load price dataset
data = pd.read_csv('data/train/cleaned_data.csv', low_memory=False)
price_map = data.groupby('material_name')['material_price'].median().to_dict()

# Load obj.names and map prices
with open('data/cvData/obj.names', 'r') as f:
    class_names = [line.strip() for line in f]
price_labels = [price_map.get(name.split()[0].lower(), data['material_price'].median()) for name in class_names]
scaler = StandardScaler()
price_labels_scaled = scaler.fit_transform(np.array(price_labels).reshape(-1, 1))
price_min = data['material_price'].min()
price_max = data['material_price'].max()
scaler_min = min(0, price_min)

# NLP setup for text embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_nlp = BertModel.from_pretrained('bert-base-multilingual-cased')
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_nlp(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

text_embeddings = np.array([get_text_embedding(name) for name in class_names])

# Define the Price Prediction Model with non-negative constraint
class PricePredictor(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(embedding_dim + 4, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu_final = nn.ReLU()  # Ensure non-negative output

    def forward(self, embeddings, box_features=None):
        if box_features is None:
            box_features = torch.zeros(embeddings.shape[0], 4)
        inputs = torch.cat((embeddings, box_features), dim=1)
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu_final(x)
        return x

# Train the Price Prediction Model
price_model = PricePredictor()
optimizer = optim.Adam(price_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop for price predictor
num_epochs = 50
text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
price_labels_tensor = torch.tensor(price_labels_scaled, dtype=torch.float32)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = price_model(text_embeddings_tensor)
    loss = criterion(outputs, price_labels_tensor)
    loss.backward()
    optimizer.step()
    print(f"Price Model Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the price model
torch.save(price_model.state_dict(), 'price_predictor.pt')
joblib.dump(scaler, 'price_scaler.joblib')

# Load YOLOv8 Nano for detection
yolo_model = YOLO('yolov8n.pt')  # Replace with 'runs/train/exp/weights/best.pt' after training
# Train YOLOv8 if not already done
# yolo train model=yolov8n.pt data=obj.data epochs=50 imgsz=640 batch=4

# Inference pipeline
def predict_tray_cost(image_path: str, yolo_model, price_model, scaler, class_names, text_embeddings) -> dict:
    # Step 1: Detect instruments with YOLOv8
    results = yolo_model.predict(image_path, imgsz=640)
    total_cost = 0
    detections = []
    
    if results and len(results) > 0 and hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            cls = int(box.cls.item())
            confidence = box.conf.item()
            name = class_names[cls]
            # Get BERT embedding for this class
            embedding = torch.tensor(text_embeddings[cls], dtype=torch.float32).unsqueeze(0)
            # Get box features (x, y, w, h)
            box_features = torch.tensor(box.xywh.cpu().numpy(), dtype=torch.float32)
            # Predict price
            price_model.eval()
            with torch.no_grad():
                price_pred_normalized = price_model(embedding, box_features)
                price_pred = scaler.inverse_transform(price_pred_normalized.cpu().numpy())[0][0] + scaler_min
                # Cap price within a reasonable range
                price_pred = max(0, min(price_pred, price_max * 2))
            total_cost += price_pred
    return {"total_cost": f"${total_cost:.2f}", "detections": detections}

# Load the price model for inference
price_model.load_state_dict(torch.load('price_predictor.pt'))
price_model.eval()

# Example usage
result = predict_tray_cost('data/cvData/obj_train_data/images/tray1.png', yolo_model, price_model, scaler, class_names, text_embeddings)
print(result)