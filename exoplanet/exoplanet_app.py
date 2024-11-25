import streamlit as st
st.set_page_config(layout="wide")

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        .element-container {
            margin-bottom: 0.5rem;
        }
        .stMarkdown {
            margin-bottom: 0rem;
        }
        div[data-testid="stSidebarContent"] > div:first-child {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

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

st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>Exoplanet Binary Classifier</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='margin-top: -1rem;'>Input Features</h3>", unsafe_allow_html=True)

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

predict_button = st.sidebar.button("Predict", use_container_width=True)

if predict_button:
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    st.markdown(f"<h4 style='text-align: center; margin-top: -1rem;'>Prediction: {prediction[0]}</h4>", unsafe_allow_html=True)
    
    dataset_path = "data/processed/cleaned_data.csv"
    data = pd.read_csv(dataset_path)

    default_blue = "#4FADFF"
    seaborn_red = "#E24A33"

    # Probability Plot with Animation
    # For the probability plot
    fig_prob.update_layout(
        title='Prediction Probabilities',
        template='plotly_dark',
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 1]),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }]
    )

    # For the radius area plot
    radius_area.update_layout(
        title='Planetary Radius Distribution',
        xaxis_title='Planetary Radius (Earth Radii)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=400,
        xaxis=dict(
            type='log',
            title_standoff=25,
            ticktext=['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'],
            tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 30},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }]
    )

    # For the planet area plot
    planet_area.update_layout(
        title='Number of Planets per System',
        xaxis_title='Number of Planets',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=400,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 30},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }]
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.plotly_chart(fig_prob, use_container_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        bins = np.logspace(np.log10(0.1), np.log10(100), 50)
        hist_data = np.histogram(data['koi_prad'], bins=bins)
        x_values = np.sqrt(bins[:-1] * bins[1:])
        
        radius_area = go.Figure()
        
        radius_area.add_trace(
            go.Scatter(
                x=x_values,
                y=np.zeros_like(hist_data[0]),
                fill='tozeroy',
                mode='lines',
                line=dict(width=2, color=default_blue),
                name='Dataset Distribution',
                hovertemplate='Radius: %{x:.2f}<br>Count: %{y}<extra></extra>'
            )
        )

        frames = [
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_values,
                        y=hist_data[0] * k/20,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=2, color=default_blue)
                    )
                ],
                name=f'frame{k}'
            )
            for k in range(21)
        ]
        radius_area.frames = frames

        user_value = user_input['koi_prad']
        bin_index = np.digitize(user_value, bins) - 1
        if 0 <= bin_index < len(hist_data[0]):
            radius_area.add_trace(
                go.Scatter(
                    x=[user_value],
                    y=[hist_data[0][bin_index]],
                    mode='markers',
                    marker=dict(size=8, color=seaborn_red),
                    name='Your Input',
                    hovertemplate='Radius: %{x:.2f}<br>Count: %{y}<extra></extra>'
                )
            )

        radius_area.update_layout(
            title='Planetary Radius Distribution',
            xaxis_title='Planetary Radius (Earth Radii)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400,
            xaxis=dict(
                type='log',
                title_standoff=25,
                ticktext=['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'],
                tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
            ),
            animation={
                'frame': {'duration': 30},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }
        )

        st.plotly_chart(radius_area, use_container_width=True)

    with col2:
        planet_count = data['koi_count'].value_counts().sort_index()
        
        planet_area = go.Figure()

        planet_area.add_trace(
            go.Scatter(
                x=planet_count.index,
                y=np.zeros_like(planet_count.values),
                fill='tozeroy',
                mode='lines',
                line=dict(width=2, color=default_blue),
                name='Dataset Distribution',
                hovertemplate='Planets: %{x}<br>Count: %{y}<extra></extra>'
            )
        )

        frames = [
            go.Frame(
                data=[
                    go.Scatter(
                        x=planet_count.index,
                        y=planet_count.values * k/20,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=2, color=default_blue)
                    )
                ],
                name=f'frame{k}'
            )
            for k in range(21)
        ]
        planet_area.frames = frames

        planet_area.add_trace(
            go.Scatter(
                x=[user_input['koi_count']],
                y=[planet_count[user_input['koi_count']]],
                mode='markers',
                marker=dict(size=8, color=seaborn_red),
                name='Your Input',
                hovertemplate='Planets: %{x}<br>Count: %{y}<extra></extra>'
            )
        )

        planet_area.update_layout(
            title='Number of Planets per System',
            xaxis_title='Number of Planets',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400,
            animation={
                'frame': {'duration': 30},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }
        )

        st.plotly_chart(planet_area, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        co_counts = data['koi_fpflag_co'].value_counts()
        co_pie = go.Figure(data=[go.Pie(
            labels=['No Offset', 'Offset'],
            values=co_counts.values,
            hole=.3,
            textinfo='percent',
            marker_colors=[default_blue, seaborn_red],
            textfont=dict(size=16, color='white'),
            textposition='inside',
            hoverinfo='label+percent+value',
            hoverlabel=dict(font=dict(size=14, color='white')),
            pull=[0, 0.1]
        )])
        
        co_pie.update_layout(
            title='Centroid Offset Distribution',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(co_pie, use_container_width=True)

    with col4:
        ss_counts = data['koi_fpflag_ss'].value_counts()
        ss_pie = go.Figure(data=[go.Pie(
            labels=['No Eclipse', 'Eclipse'],
            values=ss_counts.values,
            hole=.3,
            textinfo='percent',
            marker_colors=[default_blue, seaborn_red],
            textfont=dict(size=16, color='white'),
            textposition='inside',
            hoverinfo='label+percent+value',
            hoverlabel=dict(font=dict(size=14, color='white')),
            pull=[0, 0.1]
        )])
        
        ss_pie.update_layout(
            title='Stellar Eclipse Distribution',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(ss_pie, use_container_width=True)

with st.expander("Menu"):
    st.write("This is a placeholder for additional features.")