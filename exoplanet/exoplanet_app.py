import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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
st.set_page_config(layout="wide")
st.title("Exoplanet Binary Classifier")
st.sidebar.header("Input Features") 

# Collect user input for the 5 required features
user_input = {}
user_input["koi_score"] = st.sidebar.number_input(
    "Disposition Score",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    help="A value between 0 and 1 indicating the confidence in the KOI disposition. Higher values indicate more confidence for CANDIDATE, lower for FALSE POSITIVE. Values between 0 and 1."
)
user_input["koi_fpflag_co"] = st.sidebar.number_input(
    "Centroid Offset Flag",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    help="Indicates that the source of the signal is from a nearby star, inferred by centroid location measurements or signal strength comparison. Values between 0 and 1."
)
user_input["koi_fpflag_ss"] = st.sidebar.number_input(
    "Stellar Eclipse Flag", 
    min_value=0.0,
    max_value=1.0, 
    value=0.0,
    help="Indicates a significant secondary event or eclipse-like variability, suggesting the signal may be caused by an eclipsing binary. Values between 0 and 1."
)
user_input["koi_prad"] = st.sidebar.number_input(
    "Planetary Radius", 
    min_value=0.0, 
    value=2.26,
    help="The radius of the planet in Earth radii, calculated from the planet-star radius ratio and the stellar radius. Values between 0.08 & 200346.0."
)
user_input["koi_count"] = st.sidebar.slider(
    "Number of Planets", 
    min_value=1,
    max_value=7, 
    value=2,
    step=1,
    help="The number of planet candidates identified in a system. Values between 1 & 7."
)

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
    print(prediction)
    # Display the prediction and the probability
    st.write(f"**Prediction:** {prediction[0]}")  # Display predicted class (e.g., 'CONFIRMED' or 'FALSE POSITIVE')
    st.write(f"**Probability:** {prediction_prob[0][1]:.2f} (CONFIRMED), {prediction_prob[0][0]:.2f} (FALSE POSITIVE)")

    # Load dataset
    dataset_path = "../data/processed/cleaned_data.csv"
    data = pd.read_csv(dataset_path)

    # Apply dark mode styling to plots
    plt.style.use("dark_background")

    # Create a 2x2 grid layout for all plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Feature Visualizations with User Input", fontsize=16)
    fig.set_facecolor("#262730") 
    # 1. Pie Chart for `koi_fpflag_co`
    pie_data_co = data["koi_fpflag_co"].value_counts()
    axs[0, 0].pie(
        pie_data_co,
        labels=pie_data_co.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#1f77b4", "#ff7f0e"],
    )
    axs[0, 0].set_title("Centroid Offset Flag (koi_fpflag_co)")

    # 2. Pie Chart for `koi_fpflag_ss`
    pie_data_ss = data["koi_fpflag_ss"].value_counts()
    axs[0, 1].pie(
        pie_data_ss,
        labels=pie_data_ss.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#1f77b4", "#ff7f0e"],
    )
    axs[0, 1].set_title("Stellar Eclipse Flag (koi_fpflag_ss)")

    # 3. Log-scaled Histogram for `koi_prad`
    sns.histplot(data["koi_prad"], ax=axs[1, 0], bins=20, kde=True, color="skyblue", log_scale=(True, False))
    axs[1, 0].axvline(user_input["koi_prad"], color="red", linestyle="--", linewidth=2, label="User Input")
    axs[1, 0].set_title("Planetary Radius (koi_prad)")
    axs[1, 0].set_xlabel("Planetary Radius")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].legend()

    # 4. Bar Chart for `koi_count`
    bar_data = data["koi_count"].value_counts().sort_index()
    bars = axs[1, 1].bar(bar_data.index, bar_data.values, color="skyblue", label="Dataset")
    user_idx = user_input["koi_count"]
    for bar in bars:
        if bar.get_x() <= user_idx < bar.get_x() + bar.get_width():
            bar.set_color("red")
            bar.set_label("User Input")
    axs[1, 1].set_title("Number of Planets (koi_count)")
    axs[1, 1].set_xlabel("Number of Planets")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].legend()
    # Apply background color to all axes
    for ax in axs.flat:
        ax.set_facecolor("#262730")  # Match Streamlit's black background
        ax.tick_params(colors="white")  # White color for ticks and labels

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)

# Right: Hamburger menu placeholder
with st.expander("Menu (Click to expand)"):
    st.write("This is a placeholder for additional features.")