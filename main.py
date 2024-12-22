# main.py
import streamlit as st
from components.layout import setup_page_layout
from components.sidebar import create_sidebar
from components.predictions import show_prediction_results
from utils.data_loader import load_model_and_data

# Setup page and load data
setup_page_layout()
model, scaler, data = load_model_and_data()

# Create sidebar inputs
user_input = create_sidebar()

# Show predictions when button is clicked
if st.session_state.get('predict_clicked', False):
    show_prediction_results(model, scaler, user_input, data)
    