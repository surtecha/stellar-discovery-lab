import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open("../saved_models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("../saved_models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the full list of features in the correct order
full_feature_order = [
    "koi_score", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_period",
    "koi_time0bk", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_teq", 
    "koi_insol", "koi_model_snr", "koi_count", "koi_num_transits", "koi_tce_plnt_num", 
    "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag", "koi_disposition"
]

# Initialize input boxes and collect user input
st.title("Exoplanet Binary Classifier")
st.sidebar.header("Input Features") 

# Collect user input for the 5 required features
user_input = {}
user_input["koi_score"] = st.sidebar.number_input("koi_score", min_value=0.0, value=1.0)
user_input["koi_fpflag_co"] = st.sidebar.number_input("koi_fpflag_co", min_value=0, value=0)
user_input["koi_fpflag_ss"] = st.sidebar.number_input("koi_fpflag_ss", min_value=0, value=0)
user_input["koi_prad"] = st.sidebar.number_input("koi_prad", min_value=0.0, value=2.26)
user_input["koi_count"] = st.sidebar.number_input("koi_count", min_value=0, value=2)

# Fill the remaining features with placeholder values (0.0 for numeric features)
for feature in full_feature_order:
    if feature not in user_input:
        if feature == "koi_disposition":
            # For categorical feature "koi_disposition", you can set a default like "CONFIRMED"
            user_input[feature] = "CONFIRMED"
        else:
            # For numerical features, use a placeholder value (e.g., 0.0)
            user_input[feature] = 0.0

# Prepare the input data by creating a list with all features in the correct order
input_data = [user_input[feature] for feature in full_feature_order]

# Convert to numpy array and reshape for prediction
input_data = np.array(input_data).reshape(1, -1)

# Separate numeric features for scaling (all except the categorical feature "koi_disposition")
numeric_features = full_feature_order[:-1]  # All except the last feature
input_data_numeric = input_data[:, :len(numeric_features)]  # Select only the numeric data

# Scale the input data using the saved scaler
scaled_input = scaler.transform(input_data_numeric)

# Button to trigger prediction
if st.button("Predict"):
    # Predict the result using the trained model
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    # Display the prediction and the probability
    st.write(f"**Prediction:** {prediction[0]}")  # Display predicted class (e.g., 'CONFIRMED' or 'FALSE POSITIVE')
    st.write(f"**Probability:** {prediction_prob[0][1]:.2f} (CONFIRMED), {prediction_prob[0][0]:.2f} (FALSE POSITIVE)")
