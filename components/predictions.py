import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import full_feature_order
from components.visualizations import (
    create_animated_area_chart, 
    create_donut_chart,
    create_probability_chart,
)
from components.simulations import load_threejs_simulation

def prepare_input_data(user_input, scaler):
    input_dict = {**user_input}
    for feature in full_feature_order:
        if feature not in input_dict:
            input_dict[feature] = "CONFIRMED" if feature == "koi_disposition" else 0.0
    
    input_data = [input_dict[feature] for feature in full_feature_order]
    input_data = np.array(input_data).reshape(1, -1)
    numeric_features = full_feature_order[:-1]
    input_data_numeric = input_data[:, :len(numeric_features)]
    
    return scaler.transform(input_data_numeric)

def show_prediction_results(model, scaler, user_input, data):
    with st.spinner('Running prediction...'):
        scaled_input = prepare_input_data(user_input, scaler)
        prediction = model.predict(scaled_input)
        prediction_prob = model.predict_proba(scaled_input)
    
    st.markdown(
        f"<h2 style='text-align: center;'>Prediction: {prediction[0]}</h2>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        prob_fig = create_probability_chart(prediction_prob)
        st.pyplot(prob_fig)

    if prediction[0] == "CONFIRMED":
        st.markdown("<h3>Transit Simulation</h3>", unsafe_allow_html=True)
        with st.spinner('Generating transit simulation...'):
            load_threejs_simulation(
                koi_count=user_input['koi_count'],
                koi_prad=user_input['koi_prad']
            )

    # Wrap visualizations in dropdown if result is CONFIRMED
    if prediction[0] == "CONFIRMED":
        with st.expander("Explore Data Visualizations"):
            col1, col2 = st.columns([1, 1])
            col3, col4 = st.columns([1, 1])

            with col1:
                bins = np.logspace(np.log10(0.1), np.log10(100), 50)
                hist_data = np.histogram(data['koi_prad'], bins=bins)
                x_values = np.sqrt(bins[:-1] * bins[1:])
                
                radius_chart = create_animated_area_chart(
                    x_values=x_values,
                    y_values=hist_data[0],
                    user_value=user_input['koi_prad'],
                    title='Planetary Radius Distribution',
                    x_title='Planetary Radius (Earth Radii)',
                    y_title='Frequency',
                    log_scale=True
                )
                st.plotly_chart(radius_chart, use_container_width=True)

            with col2:
                planet_counts = data['koi_count'].value_counts().sort_index()
                planet_chart = create_animated_area_chart(
                    x_values=planet_counts.index,
                    y_values=planet_counts.values,
                    user_value=user_input['koi_count'],
                    title='Number of Planets per System',
                    x_title='Number of Planets',
                    y_title='Frequency'
                )
                st.plotly_chart(planet_chart, use_container_width=True)

            with col3:
                co_counts = data['koi_fpflag_co'].value_counts()
                co_pie = create_donut_chart(
                    values=co_counts.values,
                    labels=['No Offset', 'Offset'],
                    title='Centroid Offset Distribution'
                )
                st.plotly_chart(co_pie, use_container_width=True)

            with col4:
                ss_counts = data['koi_fpflag_ss'].value_counts()
                ss_pie = create_donut_chart(
                    values=ss_counts.values,
                    labels=['No Eclipse', 'Eclipse'],
                    title='Stellar Eclipse Distribution'
                )
                st.plotly_chart(ss_pie, use_container_width=True)
    else:
        # Show visualizations directly if result is not CONFIRMED
        col1, col2 = st.columns([1, 1])
        col3, col4 = st.columns([1, 1])

        with col1:
            bins = np.logspace(np.log10(0.1), np.log10(100), 50)
            hist_data = np.histogram(data['koi_prad'], bins=bins)
            x_values = np.sqrt(bins[:-1] * bins[1:])
            
            radius_chart = create_animated_area_chart(
                x_values=x_values,
                y_values=hist_data[0],
                user_value=user_input['koi_prad'],
                title='Planetary Radius Distribution',
                x_title='Planetary Radius (Earth Radii)',
                y_title='Frequency',
                log_scale=True
            )
            st.plotly_chart(radius_chart, use_container_width=True)

        with col2:
            planet_counts = data['koi_count'].value_counts().sort_index()
            planet_chart = create_animated_area_chart(
                x_values=planet_counts.index,
                y_values=planet_counts.values,
                user_value=user_input['koi_count'],
                title='Number of Planets per System',
                x_title='Number of Planets',
                y_title='Frequency'
            )
            st.plotly_chart(planet_chart, use_container_width=True)

        with col3:
            co_counts = data['koi_fpflag_co'].value_counts()
            co_pie = create_donut_chart(
                values=co_counts.values,
                labels=['No Offset', 'Offset'],
                title='Centroid Offset Distribution'
            )
            st.plotly_chart(co_pie, use_container_width=True)

        with col4:
            ss_counts = data['koi_fpflag_ss'].value_counts()
            ss_pie = create_donut_chart(
                values=ss_counts.values,
                labels=['No Eclipse', 'Eclipse'],
                title='Stellar Eclipse Distribution'
            )
            st.plotly_chart(ss_pie, use_container_width=True)
