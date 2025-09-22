import pandas as pd
import json

def load_csv(file_path):
    return pd.read_csv(file_path).to_dict(orient='records')

def mock_health_api(district='Ferozepur'):
    """Simulate IDSP API with real-ish CSV data."""
    data = load_csv('data/health_cases.csv')
    filtered = [d for d in data if district.lower() in d['district'].lower()]
    return filtered if filtered else data[:2]  # Default to first two rows

def mock_water_api(district='Ferozepur'):
    """Simulate CPCB API with CSV data."""
    data = load_csv('data/water_quality.csv')
    filtered = [d for d in data if district.lower() in d['district'].lower()]
    return filtered[0] if filtered else data[0]

def real_rainfall_api():
    return {'rainfall_mm': 250, 'source': 'IMD Punjab Sep 2025 (Sphere India Report)'}