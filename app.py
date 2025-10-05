import pandas as pd
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import json
from utils import load_csv, mock_health_api, mock_water_api, real_rainfall_api
import pickle
try:
    from sklearn.preprocessing import StandardScaler  # optional; required only for model/scaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    StandardScaler = None
    _SKLEARN_AVAILABLE = False
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env if present
load_dotenv()

# Configure Gemini API key securely (graceful if missing)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-2.5-flash')
else:
    model_gemini = None

# Load ML model (graceful if Torch/artifacts missing)
ml_model = None
scaler = None
if torch and nn and _SKLEARN_AVAILABLE:
    class OutbreakPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 10)
            self.fc2 = nn.Linear(10, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.sigmoid(self.fc2(x))
    try:
        ml_model = OutbreakPredictor()
        ml_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        ml_model.eval()
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print('[WARN] ML artifacts not found (model.pth/scaler.pkl). /predict will fallback.')
    except Exception as e:
        print(f'[WARN] Failed to initialize ML model/scaler: {e}. /predict will fallback.')
else:
    if not torch or not nn:
        print('[WARN] PyTorch not installed. /predict will fallback to heuristic.')
    if not _SKLEARN_AVAILABLE:
        print('[WARN] scikit-learn not installed. /predict will fallback to heuristic.')

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
    # Fallback if model/scaler unavailable
    if not ml_model or not scaler:
        avg = sum(features) / max(len(features), 1)
        prob = min(max(avg / 1000.0, 0.0), 1.0)
        risk_level = 'High' if prob > 0.5 else 'Low'
        return jsonify({'risk_prob': prob, 'risk_level': risk_level, 'note': 'heuristic (model unavailable)'})
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
    if not model_gemini:
        bot_reply = 'Chat temporarily unavailable: missing GEMINI_API_KEY. Please configure the environment.'
        return jsonify({'reply': bot_reply})
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

# Friendly 404 page
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404