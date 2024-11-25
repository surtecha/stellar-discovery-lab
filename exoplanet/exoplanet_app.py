import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

with open("saved_models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("saved_models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

full_feature_order = [
    "koi_score", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_period",
    "koi_time0bk", "koi_impact", "koi_duration", "koi_depth", "koi_prad", "koi_teq", 
    "koi_insol", "koi_model_snr", "koi_count", "koi_num_transits", "koi_tce_plnt_num", 
    "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag", "koi_disposition"
]

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Exoplanet Binary Classifier</h1>", unsafe_allow_html=True)
st.sidebar.header("Input Features") 

user_input = {}
user_input["koi_score"] = st.sidebar.number_input("Disposition Score", min_value=0.0, max_value=1.0, value=1.0)
user_input["koi_fpflag_co"] = st.sidebar.number_input("Centroid Offset Flag", min_value=0.0, max_value=1.0, value=0.0)
user_input["koi_fpflag_ss"] = st.sidebar.number_input("Stellar Eclipse Flag", min_value=0.0, max_value=1.0, value=0.0)
user_input["koi_prad"] = st.sidebar.number_input("Planetary Radius", min_value=0.0, value=2.26)
user_input["koi_count"] = st.sidebar.slider("Number of Planets", min_value=1, max_value=7, value=2, step=1)

for feature in full_feature_order:
    if feature not in user_input:
        user_input[feature] = "CONFIRMED" if feature == "koi_disposition" else 0.0

input_data = [user_input[feature] for feature in full_feature_order]
input_data = np.array(input_data).reshape(1, -1)
numeric_features = full_feature_order[:-1]
input_data_numeric = input_data[:, :len(numeric_features)]
scaled_input = scaler.transform(input_data_numeric)

st.sidebar.write("")  # Add some spacing
predict_button = st.sidebar.button("Predict", use_container_width=True)

if predict_button:
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    # Make prediction results bigger and bolder, but smaller than title
    st.markdown(f"<h4 style='text-align: center;'>Prediction: {prediction[0]}</h3>", unsafe_allow_html=True)
    
    dataset_path = "data/processed/cleaned_data.csv"
    data = pd.read_csv(dataset_path)

    # Probability Bar Chart using matplotlib
    fig_prob, ax_prob = plt.subplots(figsize=(8, 5))
    fig_prob.patch.set_facecolor('#0E1117')
    ax_prob.set_facecolor('#0E1117')

    probabilities = [prediction_prob[0][0], prediction_prob[0][1]]
    labels = ['CONFIRMED', 'FALSE POSITIVE']
    colors = ['#3498db', '#e74c3c']
    
    bars = ax_prob.bar(labels, probabilities, color=colors)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title('Prediction Probabilities', color='white', pad=15, fontsize=20) 
    
    for bar in bars:
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height/2.,
                    f'{height:.1%}',
                    ha='center', va='center', color='white', fontsize=14, weight='bold')
    
    ax_prob.tick_params(colors='white')
    ax_prob.spines['bottom'].set_color('white')
    ax_prob.spines['top'].set_color('white')
    ax_prob.spines['right'].set_color('white')
    ax_prob.spines['left'].set_color('white')
    
    for spine in ax_prob.spines.values():
        spine.set_visible(True)
        spine.set_color('white')
    
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig_prob)

    # Create 2x2 layout with equal width columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Planetary Radius Distribution
        radius_hist = go.Figure()
        
        # Create logarithmically spaced bins
        bins = np.logspace(np.log10(0.1), np.log10(100), 30)
        hist_data = np.histogram(data['koi_prad'], bins=bins)
        
        bar_colors = ['rgba(135, 206, 235, 0.7)'] * len(hist_data[0])
        user_value = user_input['koi_prad']
        bin_index = np.digitize(user_value, bins) - 1
        if 0 <= bin_index < len(bar_colors):
            bar_colors[bin_index] = 'rgba(255, 0, 0, 0.7)'

        radius_hist.add_trace(go.Bar(
            x=bins[:-1],
            y=hist_data[0],
            width=np.diff(bins),
            marker_color=bar_colors,
            name='Dataset'
        ))

        radius_hist.update_layout(
            title='Planetary Radius Distribution',
            xaxis_title='Planetary Radius (Earth Radii)',
            yaxis_title='Frequency',
            template='plotly_dark',
            showlegend=True,
            height=400,
            width=None,
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis=dict(
                range=[0, max(hist_data[0]) * 1.1]
            ),
            xaxis=dict(
                type='log',
                title_standoff=25,
                ticktext=['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'],
                tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                tickangle=0,
                gridwidth=1,
                showgrid=True,
            )
        )

        st.plotly_chart(radius_hist, use_container_width=True)

    with col2:
        # Number of Planets Distribution
        planet_count = data['koi_count'].value_counts().sort_index()
        
        bar_colors = ['rgba(135, 206, 235, 0.7)'] * len(planet_count)
        user_count_index = planet_count.index.get_loc(user_input['koi_count'])
        bar_colors[user_count_index] = 'rgba(255, 0, 0, 0.7)'

        planet_hist = go.Figure(data=[
            go.Bar(
                x=planet_count.index,
                y=planet_count.values,
                marker_color=bar_colors,
                name='Dataset'
            )
        ])

        planet_hist.update_layout(
            title='Number of Planets per System',
            xaxis_title='Number of Planets',
            yaxis_title='Frequency',
            template='plotly_dark',
            showlegend=True,
            height=400,
            width=None,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.plotly_chart(planet_hist, use_container_width=True)

    # Pie Charts
    col3, col4 = st.columns(2)

    with col3:
        # Centroid Offset Flag Pie Chart
        co_counts = data['koi_fpflag_co'].value_counts()
        co_pie = go.Figure(data=[go.Pie(
            labels=['No Offset', 'Offset'],
            values=co_counts.values,
            hole=.3,
            textinfo='percent+label'
        )])
        co_pie.update_layout(
            title='Centroid Offset Distribution',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(co_pie, use_container_width=True)

    with col4:
        # Stellar Eclipse Flag Pie Chart
        ss_counts = data['koi_fpflag_ss'].value_counts()
        ss_pie = go.Figure(data=[go.Pie(
            labels=['No Eclipse', 'Eclipse'],
            values=ss_counts.values,
            hole=.3,
            textinfo='percent+label'
        )])
        ss_pie.update_layout(
            title='Stellar Eclipse Distribution',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(ss_pie, use_container_width=True)

with st.expander("Menu"):
    st.write("This is a placeholder for additional features.")