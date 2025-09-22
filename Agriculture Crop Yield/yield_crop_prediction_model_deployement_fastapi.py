from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize FastAPI app with metadata
app = FastAPI(
    title= "Deployed a Crop Yield Prediction model using FastAPI",
    description= "Deployed a Crop Yield Prediction model using FastAPI. The project integrates machine learning with a scalable API for real-time predictions, enabling farmers and researchers to estimate crop yield efficiently with accessible endpoints.",
    version= "1.0.0"
)

# Load trained model and preprocessing objects
try:
    model = load_model("crop_model.keras")
    with open("scale.pkl", "rb") as file:
        scale_loaded = pickle.load(file)
    with open("le.pkl", "rb") as file:
        le_loaded = pickle.load(file)

except FileNotFoundError as e:
    print(f"Some file is not Found: {e}")

class CropInputs(BaseModel):
    Region: str
    Soil_Type: str
    Crop: str
    Rainfall_mm: float
    Temperature_Celsius: float
    Fertilizer_Used: int
    Irrigation_Used: int
    Weather_Condition: str
    Days_to_Harvest: int

# Define response schema for consistent API output
class PredictionResponse(BaseModel):
    predicted_value: float
    crop_features: dict

cat_cols = ["Region", "Crop", "Soil_Type", "Weather_Condition"]
num_cols = ["Rainfall_mm", "Temperature_Celsius", "Fertilizer_Used", "Irrigation_Used", "Days_to_Harvest"]


def preprocess_input(data: dict):
    # Encode categorical features
    cat_features = []
    for col in cat_cols:
        le = le_loaded[col]
        try:
            val = le.transform([data[col]])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category for '{col}': {data[col]}")
        cat_features.append(np.array([[val]]))  # <-- shape (1,1)

    # Scale numeric features ONLY
    num_values = np.array([data[col] for col in num_cols]).reshape(1, -1)
    num_scaled = scale_loaded.transform(num_values)  # shape (1,5)

    return cat_features + [num_scaled]

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Crop Prediction Model! Go to /docs for Swagger UI."}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def prediction_func(crop: CropInputs):
    try:
        model_input = preprocess_input(crop.dict())
        prediction = model.predict(model_input)[0][0]

        return PredictionResponse(
            predicted_value= prediction,
            crop_features= crop.dict()
        )
    except Exception as e:
       # If something fails, return 500 error
       raise HTTPException(status_code=500, detail= f"Something is going wrong. {str(e)}")
    
# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host= "127.0.0.1", port= 8000)

