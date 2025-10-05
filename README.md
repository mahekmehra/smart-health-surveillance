# Smart Health Surveillance System - Punjab Floods 2025

A prototype web application for monitoring and predicting water-borne disease outbreaks (e.g., cholera, typhoid, diarrhea, hepatitis) in Punjab, India, during the 2025 floods. Built for a hackathon, it features community symptom reporting, AI-driven outbreak prediction, simulated API data fetching, a multilingual chatbot for health education, and visualization through dashboards and alerts. The system aims to support health officials and communities in flood-affected areas like Ferozepur.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Collaboration](#collaboration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Overview
This prototype addresses the health crisis caused by the Punjab Floods 2025, affecting ~2 million people across 23 districts. It simulates a real-time health surveillance system with:
- **Community Reporting**: Collects symptom data from users (e.g., ASHA workers).
- **AI Prediction**: Uses a PyTorch neural network to predict outbreak risks.
- **Simulated APIs**: Fetches mock health and water quality data.
- **Multilingual Chatbot**: Educates users in English, Hindi, and Punjabi using Google's Gemini API.
- **Dashboard**: Visualizes flood impacts and disease hotspots.
- **Alerts**: Provides outbreak warnings in multiple languages.

The app runs locally using Flask and is designed for scalability with real APIs and databases in production.

## Features
1. **Report Symptoms**:
   - Users submit symptoms, locations, and case counts via a form.
   - Data saved to `data/reports.csv` for analysis.
   - Example: Report "diarrhea, Ferozepur, 5 cases".

2. **Outbreak Prediction**:
   - AI model predicts outbreak risk (Low/High) based on case counts (diarrhea, typhoid, cholera, hepatitis).
   - Uses a PyTorch neural network trained on `data/health_cases.csv`.
   - Example: Input [5000, 100, 50, 100] → "Low risk"; [16000, 460, 160, 310] → "High risk".

3. **API Fetch (Simulated)**:
   - Mock APIs for health (`/api/health`) and water quality (`/api/water`) data.
   - Simulates IDSP (health) and CPCB (water) data for districts like Ferozepur.

4. **Multilingual Chatbot**:
   - Powered by Google's Gemini API.
   - Answers health queries in English, Hindi, or Punjabi using flood data.
   - Example: "What are cholera symptoms?" in Punjabi.

5. **Dashboard**:
   - Displays flood statistics (e.g., 2M affected, 23 districts) and hardcoded hotspots.
   - Includes a placeholder map (`static/images/punjab_map.png`).

6. **Alerts**:
   - Shows outbreak warnings (e.g., "Ferozepur: 150 cholera cases") in English, Hindi, or Punjabi.
   - Uses `data/health_cases.csv` and `mock_data.json`.

## Project Structure
```
health_surveillance_prototype/
├── data/
│   ├── flood_impacts.csv      # Flood statistics (2M affected, 23 districts)
│   ├── health_cases.csv       # Disease case data (diarrhea, cholera, etc.)
│   ├── reports.csv            # User-submitted symptom reports
│   ├── water_quality.csv      # Water quality data (pH, turbidity, coliform)
├── static/
│   ├── css/style.css          # Styling for forms, tables, etc.
│   ├── images/                # Placeholder images (hotspot_map.png, punjab_map.png)
│   ├── js/script.js           # JavaScript for AJAX form submissions
├── templates/
│   ├── alerts.html            # Alerts page template
│   ├── dashboard.html         # Dashboard page template
│   ├── index.html             # Main page with forms and chatbot
├── .gitignore                 # Ignores venv/, *.pth, *.pkl, __pycache__/
├── app.py                     # Flask app with routes and logic
├── ml_model.py                # Trains PyTorch model, saves model.pth, scaler.pkl
├── mock_data.json             # Mock translations and alerts
├── requirements.txt           # Python dependencies
├── utils.py                   # Helper functions for data loading and APIs
```

## Dependencies
- Python 3.13
- Libraries (listed in `requirements.txt`):
  ```
  flask==3.0.3
  google-generativeai==0.8.3
  torch==2.8.0
  requests==2.32.4
  pandas==2.2.2
  scikit-learn==1.5.2
  python-dotenv==1.0.1
  ```

## Usage
- **Homepage (`/`)**: Access all features via forms and links.
  - **Report Symptoms**: Enter symptom (e.g., "diarrhea"), location (e.g., "Ferozepur"), and case count.
  - **Outbreak Prediction**: Input case counts (e.g., diarrhea 5000, typhoid 100, cholera 50, hepatitis 100) to predict risk.
  - **API Fetch**: Click buttons to fetch mock health/water data for Ferozepur.
  - **Chatbot**: Select language (English/Hindi/Punjabi), ask health questions (e.g., "cholera prevention").
- **Dashboard (`/dashboard`)**: View flood statistics and hotspots.
- **Alerts (`/alerts?lang=english`)**: See outbreak warnings (change `lang` to `hindi` or `punjabi`).

## Data Sources
- **flood_impacts.csv**: Sphere India Secondary Data Analysis Report (ReliefWeb, Sep 2025).
- **health_cases.csv**: Proxy from NIH IDSR Week 9 2025 (Pakistan), adjusted +20% for floods.
- **water_quality.csv**: Historical CPCB NWMP data (2020-2024), adjusted for flood conditions.
- **mock_data.json**: Custom translations for alerts and health education.
- **reports.csv**: Dynamically generated from user reports.


## Future Enhancements
- Integrate real-time APIs (IDSP, CPCB, IMD).
- Add Chart.js for dynamic visualizations.
- Use Folium/Leaflet for interactive maps.
- Deploy to Render/Heroku for live demo.
- Add authentication for secure reporting.
- Expand chatbot with more health data.