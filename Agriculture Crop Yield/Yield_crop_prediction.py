import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Load dataset ---
crop = pd.read_csv("crop_yield.csv")
crop = crop.drop_duplicates()  # remove duplicates

# Ensure binary columns are integers
crop["Fertilizer_Used"] = crop["Fertilizer_Used"].astype(int)
crop["Irrigation_Used"] = crop["Irrigation_Used"].astype(int)

# --- Scale numeric features ---
num_col = crop.select_dtypes(exclude="object").columns.drop("Yield_tons_per_hectare")
scaling = StandardScaler()
crop[num_col] = scaling.fit_transform(crop[num_col])

# --- Encode categorical columns ---
embd_cols = ["Region", "Crop", "Soil_Type", "Weather_Condition"]
encoders = {}
for col in embd_cols:
    emb_le = LabelEncoder()
    crop[col] = emb_le.fit_transform(crop[col])
    encoders[col] = emb_le

# --- Remove outliers using Z-score ---
target = "Yield_tons_per_hectare"
num_cols = crop.select_dtypes(include=[np.number]).columns.drop(target)
num_cols = [col for col in num_cols if crop[col].nunique() > 2]
z_score = np.abs(stats.zscore(crop[num_cols]))
cno = crop[(z_score < 3).all(axis=1)]  # cleaned dataset

print("Original shape:", crop.shape)
print("After outlier removal:", cno.shape)

# --- Build embedding layers for categorical features ---
emb_input, emb_layer = [], []
emb_cols = ["Region", "Crop", "Soil_Type", "Weather_Condition"]
for col in emb_cols:
    vocab_size = cno[col].nunique() + 1
    emb_dim = min(100, vocab_size // 2)

    inputs = Input(shape=(1,), name=f"{col}_input")
    emb = Embedding(input_dim=vocab_size, output_dim=emb_dim, name=f"{col}_emb")(inputs)
    emb = Flatten()(emb)

    emb_input.append(inputs)
    emb_layer.append(emb)

# --- Numeric features input ---
num_columns = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Days_to_Harvest",
]
num_inputs = Input(shape=(len(num_columns),), name="numeric_columns")

# Combine embeddings + numeric
x = Concatenate()(emb_layer + [num_inputs])

# --- Dense layers ---
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="linear")(x)  # regression output

# --- Build & compile model ---
all_inputs = emb_input + [num_inputs]
model = Model(inputs=all_inputs, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])

# --- Train-test split ---
X = cno.drop(columns=target)
y = cno[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare inputs for model
X_train_inputs = [X_train[col].values for col in emb_cols] + [X_train[num_columns].values]
X_test_inputs = [X_test[col].values for col in emb_cols] + [X_test[num_columns].values]

# --- Training with early stopping ---
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(X_train_inputs, y_train, epochs=200, batch_size=32,
          validation_split=0.2, callbacks=[early_stopping])

# --- Predictions ---
y_pred_train = model.predict(X_train_inputs).flatten()
y_pred_test = model.predict(X_test_inputs).flatten()

# --- Evaluation ---
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Results
print("Train MAE:", mae_train)
print("Train MSE:", mse_train)
print("Train R²:", r2_train)
print("Test MAE:", mae_test)
print("Test MSE:", mse_test)
print("Test R²:", r2_test)

import pickle
model_filename = "crop_model.keras"
scale_filename = "scale.pkl"
le_filename = "le.pkl"

model.save(model_filename) # model

with open(scale_filename, "wb") as file:
    pickle.dump(scaling, file) # scaler
with open(le_filename, "wb") as file:
    pickle.dump(encoders, file)     # label encoder

print("All files saved successfully !")

