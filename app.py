import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import json
from utils import load_csv, mock_health_api, mock_water_api, real_rainfall_api
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
genai.configure(api_key='AIzaSyDb0DwXDbWeruX1Rh1HcnDzYkDSLheCOdo')
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load ML model
class OutbreakPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

ml_model = OutbreakPredictor()
ml_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
ml_model.eval()

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data
flood_stats = load_csv('data/flood_impacts.csv')
with open('mock_data.json', 'r', encoding='utf-8') as f:
    mock_data = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    hotspots = [
        {'area': 'Ferozepur', 'risk': 'High', 'disease': 'Cholera', 'cases': 150},
        {'area': 'Amritsar', 'risk': 'Medium', 'disease': 'Typhoid', 'cases': 400}
    ]
    return render_template('dashboard.html', hotspots=hotspots, flood_stats=flood_stats)

@app.route('/alerts')
def alerts():
    lang = request.args.get('lang', 'english')
    health_data = load_csv('data/health_cases.csv')
    alerts = [{'message': f"{row['district']}: {row['cholera_cases']} cholera cases (Source: {row['source']})"} for row in health_data]
    alerts += [{'message': alert.get(lang, alert['message'])} for alert in mock_data['alerts']]
    return render_template('alerts.html', alerts=alerts)

@app.route('/report', methods=['POST'])
def report():
    data = request.json
    df = pd.read_csv('data/reports.csv')
    new_report = pd.DataFrame([data])
    df = pd.concat([df, new_report], ignore_index=True)
    df.to_csv('data/reports.csv', index=False)
    return jsonify({'status': 'success'})

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']  # [diarrhea, typhoid, cholera, hepatitis]
    features_normalized = scaler.transform([features])  # Normalize
    input_tensor = torch.tensor(features_normalized).float()
    with torch.no_grad():
        pred = ml_model(input_tensor).item()
    risk_level = 'High' if pred > 0.5 else 'Low'
    return jsonify({'risk_prob': pred, 'risk_level': risk_level})

@app.route('/api/health')
def api_health():
    district = request.args.get('district', 'Ferozepur')
    return jsonify(mock_health_api(district))

@app.route('/api/water')
def api_water():
    district = request.args.get('district', 'Ferozepur')
    return jsonify(mock_water_api(district))

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    lang = request.json['lang']
    flood_data = load_csv('data/flood_impacts.csv')
    prompt = f"You are a health education chatbot for Punjab floods 2025. Use real data: {flood_data}. Respond in {lang.capitalize()}. Query: {user_message}"
    try:
        response = model_gemini.generate_content(
            prompt,
            safety_settings=[
                {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
            ]
        )
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Error with Gemini API: {str(e)}. Please check API key or try again later."
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)