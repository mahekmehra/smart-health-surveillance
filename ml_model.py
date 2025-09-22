import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle

# Define the model
class OutbreakPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# Load data
data = pd.read_csv('data/health_cases.csv')
features = data[['diarrhea_cases', 'typhoid_cases', 'cholera_cases', 'hepatitis_cases']].values

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create synthetic labels (high risk if diarrhea > 10000 or cholera > 100)
labels = ((data['diarrhea_cases'] > 10000) | (data['cholera_cases'] > 100)).astype(int).values

# Convert to tensors
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

# Initialize model, loss, optimizer
model = OutbreakPredictor()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
print("Model trained on health_cases.csv and saved to model.pth")