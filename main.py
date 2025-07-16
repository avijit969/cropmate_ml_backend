from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("soil_model.pkl")
scaler = joblib.load("soil_scaler.pkl")

model2 = joblib.load("crop_model.pkl")
scaler2 = joblib.load("crop_scaler.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# FastAPI instance
app = FastAPI(title="Soil Fertility Prediction API")

# Define expected input
class SoilFeatures(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float

class CropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict-soil-fertility")
def predict_soil_fertility(data: SoilFeatures):
    features = np.array([[data.N, data.P, data.K, data.pH, data.EC, data.OC,data.S, data.Zn, data.Fe, data.Cu, data.Mn, data.B]])
    
    # Apply same scaling
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return {
        "prediction": int(prediction),
        "message": "Fertile" if prediction == 1 else "Not Fertile"
    }
@app.post("/predict-crop")
def predict_crop(features: CropFeatures):
    data = np.array([[features.N, features.P, features.K, features.temperature,features.humidity, features.ph, features.rainfall]])
    
    # Scale input
    data_scaled = scaler2.transform(data)
    
    # Predict encoded label
    pred_encoded = model2.predict(data_scaled)[0]
    
    # Decode label
    pred_crop = label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "predicted_crop": pred_crop
    }