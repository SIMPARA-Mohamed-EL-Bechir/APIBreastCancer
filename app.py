import os
from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Charger le modèle enregistré
model = joblib.load('best_model_logreg.pkl')

@app.route('/')
def home():
    return "API de prédiction du risque de cancer. Utilisez l'endpoint /predict pour faire des prédictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    required_fields = ['menopaus', 'agegrp', 'density', 'race', 
                       'bmi', 'agefirst', 'nrelbc', 'hrt']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    features = np.array([data['menopaus'], data['agegrp'], data['density'], data['race'], 
                         data['bmi'], data['agefirst'], data['nrelbc'], data['hrt']]).reshape(1, -1)

    probabilities = model.predict_proba(features)
    cancer_probability = probabilities[0][1]

    return jsonify({
        'probability_cancer': float(cancer_probability),
        'prediction': int(model.predict(features)[0])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
