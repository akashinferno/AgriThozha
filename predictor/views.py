from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import pickle
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
crop_model_path = os.path.join(BASE_DIR, 'models', 'crop_recommendationr.pkl')
fertilizer_model_path = os.path.join(BASE_DIR, 'models', 'fertilizer_recommendation.pkl')

crop_scaler_path = os.path.join(BASE_DIR, 'models', 'minmax_scaler.pkl')
fertilizer_scaler_path = os.path.join(BASE_DIR, 'models', 'standard_scalerFR.pkl')

# ---------- Load Models ----------
with open(crop_model_path, 'rb') as file:
    crop_model = pickle.load(file)

with open(fertilizer_model_path, 'rb') as file:
    fertilizer_model = pickle.load(file)

# ---------- Load Scalers ----------
crop_scaler = None
fertilizer_scaler = None

if os.path.exists(crop_scaler_path):
    with open(crop_scaler_path, 'rb') as file:
        crop_scaler = pickle.load(file)

if os.path.exists(fertilizer_scaler_path):
    with open(fertilizer_scaler_path, 'rb') as file:
        fertilizer_scaler = pickle.load(file)


# ---------- CROP PREDICTION ----------
@api_view(['POST'])
def predict_crop(request):
    try:
        data = request.data

        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return Response({'error': f'Missing fields: {missing}'}, status=400)

        nitrogen = float(data.get('nitrogen'))
        phosphorus = float(data.get('phosphorus'))
        potassium = float(data.get('potassium'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))

        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        if crop_scaler:
            features = crop_scaler.transform(features)

        prediction = crop_model.predict(features)[0]

        return Response({'recommended_crop': str(prediction)})

    except Exception as e:
        return Response({'error': str(e)}, status=400)


# ---------- FERTILIZER PREDICTION ----------
@api_view(['POST'])
def predict_fertilizer(request):
    try:
        data = request.data

        required_fields = [
            'temperature', 'humidity', 'moisture',
            'soil_type', 'crop_type',
            'nitrogen', 'potassium', 'phosphorous'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return Response({'error': f'Missing fields: {missing}'}, status=400)

        # Extract numeric fields
        nitrogen = float(data.get('nitrogen'))
        phosphorous = float(data.get('phosphorous'))
        potassium = float(data.get('potassium'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        moisture = float(data.get('moisture'))

        # Encode categorical fields (MUST match your training encoding)
        soil_map = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}
        crop_map = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Sugarcane': 3, 'Cotton': 4}

        soil_type = soil_map.get(data.get('soil_type'))
        crop_type = crop_map.get(data.get('crop_type'))

        if soil_type is None or crop_type is None:
            return Response({'error': 'Invalid soil_type or crop_type value'}, status=400)

        # ✅ Maintain the same order as training
        features = np.array([[nitrogen, phosphorous, potassium,
                              temperature, humidity, moisture,
                              soil_type, crop_type]])

        # Scale if scaler exists
        if fertilizer_scaler:
            features = fertilizer_scaler.transform(features)

        # Predict
        prediction = fertilizer_model.predict(features)[0]

        # Fertilizer label mapping (adjust if needed)
        fertilizer_labels = {
            0: "Urea",
            1: "DAP",
            2: "MOP",
            3: "10-26-26",
            4: "14-35-14",
            5: "17-17-17",
            6: "20-20",
            7: "28-28",
            8: "Ammonium Sulphate",
            9: "Ammonium Chloride"
        }

        fertilizer_name = fertilizer_labels.get(int(prediction), "Unknown Fertilizer")

        return Response({'recommended_fertilizer': fertilizer_name})

    except Exception as e:
        return Response({'error': str(e)}, status=400)

