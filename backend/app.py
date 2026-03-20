import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)