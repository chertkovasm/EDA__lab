from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from eda import transform_data

app = Flask(__name__)

model = joblib.load("/app/trained_model/trained_model.pkl")
scaler = joblib.load("/app/trained_model/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    df = pd.DataFrame(content, index=[0])
    
    if 'Rings' in df.columns:
        df = df.drop(['Rings'], axis=1)
    
    df = transform_data(df)
    
    X_scaled = scaler.transform(df)
    result = model.predict(X_scaled)[0]
    
    return jsonify({'predicted_age': float(result)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)