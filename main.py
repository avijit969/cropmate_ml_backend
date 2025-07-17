from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler for soil fertility prediction
model = joblib.load("models/fertility/soil_model.pkl")
scaler = joblib.load("models/fertility/soil_scaler.pkl")

# Load model and scaler for crop prediction
model2 = joblib.load("models/crop_type/crop_model.pkl")
scaler2 = joblib.load("models/crop_type/crop_scaler.pkl")
label_encoder = joblib.load("models/crop_type/crop_label_encoder.pkl")

# Load the model and scaler for crop prediction with location
scaler3 = joblib.load("models/crop_type_location/crop_scaler_svm.pkl")
clf = joblib.load("models/crop_type_location/dag_svm_model.pkl")
crop_le = joblib.load("models/crop_type_location/crop_label_encoder_svm.pkl")
category_encoders = joblib.load("models/crop_type_location/category_encoders_svm.pkl")

# FastAPI instance
app = FastAPI(title="CropMate ML API")

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

class CropInput(BaseModel):
    Latitude: float
    Longitude: float
    District: str
    Area: str
    Season: str
    Temperature: float
    Moisture: float
    Water_Level: float
    pH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    EC: float
    OC: float

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

@app.post("/predict-crop-location")
def predict_crop(data: CropInput):
    # Encode categorical fields
    try:
        district_enc = category_encoders['District'].transform([data.District])[0]
        area_enc = category_encoders['Area'].transform([data.Area])[0]
        season_enc = category_encoders['Season'].transform([data.Season])[0]
    except Exception as e:
        return {"error": f"Encoding error: {str(e)}"}

    # Prepare input array
    features = np.array([[
        data.Latitude, data.Longitude, district_enc, area_enc, season_enc,
        data.Temperature, data.Moisture, data.Water_Level, data.pH,
        data.Nitrogen, data.Phosphorous, data.Potassium,
        data.EC, data.OC
    ]])
    features_scaled = scaler3.transform(features)

    pred = clf.predict(features_scaled)[0]
    prob = clf.predict_proba(features_scaled).max()
    crop = crop_le.inverse_transform([pred])[0]

    return {
        "predicted_crop": crop,
        "confidence": round(prob, 4)
    }