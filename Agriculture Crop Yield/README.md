
Crop Yield Prediction

This project predicts crop yield using a deep learning model and deploys it with FastAPI for real-time forecasting. It leverages categorical embeddings (Region, Soil, Crop, Weather) and scaled numeric features (Rainfall, Temperature, Fertilizer, Irrigation, Days to Harvest) to deliver accurate predictions.

📌 Features

Hybrid deep learning model with embeddings + dense layers

Encodes categorical and numeric agricultural factors

Real-time prediction API powered by FastAPI

Interactive Swagger UI at /docs

Scalable and reusable for agricultural research

📊 Example Input (JSON)
{
  "Region": "North",
  "Soil_Type": "Loam",
  "Crop": "Barley",
  "Rainfall_mm": 200,
  "Temperature_Celsius": 22,
  "Fertilizer_Used": 1,
  "Irrigation_Used": 0,
  "Weather_Condition": "Sunny",
  "Days_to_Harvest": 120
}

📈 Example Output
{
  "predicted_value": 6.23,
  "crop_features": {
    "Region": "North",
    "Soil_Type": "Loam",
    "Crop": "Barley",
    "Rainfall_mm": 200,
    "Temperature_Celsius": 22,
    "Fertilizer_Used": 1,
    "Irrigation_Used": 0,
    "Weather_Condition": "Sunny",
    "Days_to_Harvest": 120
  }
}

📊 Model Results

The model was trained with early stopping and evaluated on both training and test sets.

Train MAE: 0.40

Train MSE: 0.25

Train R²: 0.91

Test MAE: 0.40

Test MSE: 0.25

✅ These results show the model generalizes well, with high accuracy and minimal error.

🌱 Use Cases

Farmers can forecast yield in advance for better planning

Agricultural researchers can analyze the impact of soil & weather conditions

Policymakers can optimize resource allocation and food supply planning

✨ This project highlights how AI in agriculture can transform farming decisions with real-time, data-driven insights.