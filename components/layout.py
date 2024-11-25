# components/layout.py
import streamlit as st

def setup_page_layout():
    st.set_page_config(layout="wide")
    
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
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                gap: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>Exoplanet Binary Classifier</h1>", 
                unsafe_allow_html=True)