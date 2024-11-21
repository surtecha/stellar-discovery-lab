import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
with open("saved_models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("saved_models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature names and their ranges
features = [
    "koi_score", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
    "koi_count", "koi_num_transits", "koi_tce_plnt_num", "koi_steff",
    "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag"
]

# Min and max values for each feature
feature_ranges = {
    "koi_score": (0.0, 1.0),
    "koi_fpflag_ss": (0.0, 1.0),
    "koi_fpflag_co": (0.0, 1.0),
    "koi_fpflag_ec": (0.0, 1.0),
    "koi_period": (0.299698, 1071.232624),
    "koi_time0bk": (120.515914, 1472.522306),
    "koi_impact": (0.0, 100.806),
    "koi_duration": (0.1046, 138.54),
    "koi_depth": (0.8, 1541400.0),
    "koi_prad": (0.08, 200346.0),
    "koi_teq": (92.0, 14667.0),
    "koi_insol": (0.02, 10947550.0),
    "koi_model_snr": (0.0, 9054.7),
    "koi_count": (1.0, 7.0),
    "koi_num_transits": (0.0, 2664.0),
    "koi_tce_plnt_num": (1.0, 8.0),
    "koi_steff": (2661.0, 15896.0),
    "koi_slogg": (0.047, 5.283),
    "koi_smet": (-2.5, 0.56),
    "koi_srad": (0.116, 229.908),
    "koi_smass": (0.094, 3.686),
    "koi_kepmag": (6.966, 20.003)
}

# Initialize sliders and collect user input
st.title("Exoplanet Binary Classifier")
st.sidebar.header("Input Features")

input_data = []
for feature in features:
    min_val, max_val = feature_ranges[feature]
    value = st.sidebar.slider(
        f"{feature}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),  # Default value as the midpoint
        step=float((max_val - min_val) / 100)  # Step size
    )
    input_data.append(value)

# Scale input data using the pre-trained scaler
input_data = np.array(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    # Display prediction result
    st.write(f"**Prediction:** {prediction[0]}")  # Outputs text labels
    st.write(f"**Probability:** {prediction_prob[0][1]:.2f} (Positive), {prediction_prob[0][0]:.2f} (Negative)")
