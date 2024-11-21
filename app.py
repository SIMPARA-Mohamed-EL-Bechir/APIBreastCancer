import numpy as np
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permet les requêtes CORS depuis n'importe quelle origine

# Charger le modèle enregistré
model = joblib.load('best_model_logreg.pkl')

@app.route('/')
def home():
    return "API de prédiction du risque de cancer. Utilisez l'endpoint /predict pour faire des prédictions."

@app.route('/predict', methods=['POST'])
def predict():
    # Extraire les données envoyées dans la requête POST
    data = request.get_json()
    
    # Vérifier que toutes les données nécessaires sont présentes
    required_fields = ['menopaus', 'agegrp', 'density', 'race', 
                       'bmi', 'agefirst', 'nrelbc', 'hrt']
    
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    # Extraire les caractéristiques de l'utilisateur et les transformer en tableau numpy
    features = np.array([data['menopaus'], data['agegrp'], data['density'], data['race'], 
                         data['bmi'], data['agefirst'], data['nrelbc'], data['hrt']]).reshape(1, -1)

    # Obtenir les probabilités avec predict_proba
    probabilities = model.predict_proba(features)
    
    # La probabilité de la classe "cancer" (1)
    cancer_probability = probabilities[0][1]
    
    # Retourner la probabilité au format JSON
    return jsonify({
        'probability_cancer': float(cancer_probability),
        'prediction': int(model.predict(features)[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)