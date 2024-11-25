# components/sidebar.py
import streamlit as st

def create_sidebar():
    st.sidebar.markdown("<h3 style='margin-top: -1rem;'>Input Features</h3>", 
                       unsafe_allow_html=True)
    
    user_input = {
        "koi_score": st.sidebar.number_input("Disposition Score", 
                                           min_value=0.0, max_value=1.0, value=1.0),
        "koi_fpflag_co": st.sidebar.number_input("Centroid Offset Flag", 
                                                min_value=0.0, max_value=1.0, value=0.0),
        "koi_fpflag_ss": st.sidebar.number_input("Stellar Eclipse Flag", 
                                                min_value=0.0, max_value=1.0, value=0.0),
        "koi_prad": st.sidebar.number_input("Planetary Radius", 
                                          min_value=0.0, value=2.26),
        "koi_count": st.sidebar.slider("Number of Planets", 
                                     min_value=1, max_value=7, value=2, step=1)
    }
    
    if st.sidebar.button("Predict", use_container_width=True):
        st.session_state.predict_clicked = True
    else:
        st.session_state.predict_clicked = False
    
    return user_input