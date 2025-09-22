import torch
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

class OutbreakPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# Load model and scaler
model = OutbreakPredictor()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Test inputs
test_inputs = [
    [5000, 100, 50, 100],  # Low risk
    [16000, 460, 160, 310]  # High risk
]
for input_data in test_inputs:
    input_normalized = scaler.transform([input_data])
    input_tensor = torch.tensor(input_normalized).float()
    with torch.no_grad():
        pred = model(input_tensor).item()
    risk = 'High' if pred > 0.5 else 'Low'
    print(f"Input: {input_data}, Prob: {pred:.4f}, Risk: {risk}")