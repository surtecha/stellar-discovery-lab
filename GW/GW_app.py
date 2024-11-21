import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import threading
import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json
from copy import deepcopy
import base64
from helper import generate_audio_stream
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("agg")

render_lock = threading.Lock()

# Page configuration
app_name = 'Signal Explorer'
st.set_page_config(page_title=app_name, page_icon=":telescope:", layout="wide")

# Define theme colors for consistent styling
THEME_COLORS = {
    'background': 'rgba(25, 25, 25, 0.0)',
    'plot_gridcolor': 'rgba(128, 128, 128, 0.2)',
    'plot_linecolor': 'rgba(128, 128, 128, 0.2)',
    'text_color': '#ffffff'
}

# Plot layout template
PLOT_TEMPLATE = {
    'layout': go.Layout(
        paper_bgcolor=THEME_COLORS['background'],
        plot_bgcolor=THEME_COLORS['background'],
        font={'color': THEME_COLORS['text_color']},
        xaxis={
            'gridcolor': THEME_COLORS['plot_gridcolor'],
            'zerolinecolor': THEME_COLORS['plot_linecolor']
        },
        yaxis={
            'gridcolor': THEME_COLORS['plot_gridcolor'],
            'zerolinecolor': THEME_COLORS['plot_linecolor']
        }
    )
}

observation_points = ['H1', 'L1', 'V1']

st.title('Astronomical Signal Explorer')

st.markdown("""
 * Navigate using the sidebar controls to configure analysis parameters
 * Visualizations will be displayed in the main panel
""")

@st.cache_data(max_entries=5)
def fetch_signal(timestamp, station, sample_rate=4096):
    signal_data = TimeSeries.fetch_open_data(station, timestamp-14, timestamp+14, sample_rate=sample_rate, cache=False)
    return signal_data

@st.cache_data(max_entries=10)
def get_signal_catalog():
    all_signals = datasets.find_datasets(type='events')
    signal_set = set()
    for signal in all_signals:
        designation = fetch_event_json(signal)['events'][signal]['commonName']
        if designation[0:2] == 'GW':
            signal_set.add(designation)
    catalog = list(signal_set)
    catalog.sort()
    return catalog

def create_timeseries_plot(signal_data, title):
    """Create an interactive time series plot using Plotly"""
    fig = go.Figure(layout=PLOT_TEMPLATE['layout'])
    
    fig.add_trace(go.Scatter(
        x=signal_data.times.value,
        y=signal_data.value,
        mode='lines',
        name='Signal',
        line=dict(color='#00ff00', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Strain',
        showlegend=False,
        height=400,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

# Sidebar configuration
st.sidebar.markdown("## Observatory and Time Selection")

signal_catalog = get_signal_catalog()

time_mode = st.sidebar.selectbox('Select Time Reference',
                                ['Catalog ID', 'Timestamp'])

if time_mode == 'Timestamp':
    time_input = st.sidebar.text_input('Reference Time', '1126259462.4')
    ref_time = float(time_input)

    st.sidebar.markdown("""
    Notable timestamps:
    * 1126259462.4    (First Detection) 
    * 1187008882.4    (Binary Merger) 
    * 1128667463.0    (Calibration Signal) 
    * 1132401286.33   (Signal Anomaly) 
    """)

else:
    selected_signal = st.sidebar.selectbox('Select Signal', signal_catalog)
    ref_time = datasets.event_gps(selected_signal)
    observation_points = list(datasets.event_detectors(selected_signal))
    observation_points.sort()
    st.subheader(selected_signal)
    st.write('Reference Time:', ref_time)
    
    try:
        signal_info = fetch_event_json(selected_signal)
        col1, col2 = st.columns(2)
        
        for id, details in signal_info['events'].items():
            with col1:
                st.write('Primary Mass:', details['mass_1_source'], 'M$_{\odot}$')
                st.write('Secondary Mass:', details['mass_2_source'], 'M$_{\odot}$')
            with col2:
                st.write('Combined Signal Strength:', int(details['network_matched_filter_snr']))
                info_url = f'https://gwosc.org/eventapi/html/event/{selected_signal}'
                st.markdown(f'Detailed Analysis: {info_url}')
    except:
        pass

selected_observatory = st.sidebar.selectbox('Observatory', observation_points)

base_rate = 4096
max_frequency = 1200
enable_high_resolution = st.sidebar.checkbox('Enable High Resolution')
if enable_high_resolution:
    base_rate = 16384
    max_frequency = 2000

# Analysis parameters
st.sidebar.markdown('## Analysis Parameters')
time_window = st.sidebar.slider('Analysis Window (seconds)', 0.1, 8.0, 1.0)
half_span = time_window / 2.0

st.sidebar.markdown('#### Signal Processing Options')
noise_reduction = st.sidebar.checkbox('Apply Noise Reduction', value=True)
frequency_bounds = st.sidebar.slider('Frequency Range (Hz)', min_value=10, max_value=max_frequency, value=(30,400))

st.sidebar.markdown('#### Transform Visualization')
intensity_cap = st.sidebar.slider('Maximum Intensity', 10, 500, 25)
resolution_factor = st.sidebar.slider('Resolution Factor', 5, 120, 5)
resolution_range = (int(resolution_factor*0.8), int(resolution_factor*1.2))

# Data retrieval and processing
status = st.text('Retrieving data...')
try:
    raw_signal = fetch_signal(ref_time, selected_observatory, base_rate)
except:
    st.warning(f'No data available from {selected_observatory} at time {ref_time}. Please select different parameters.')
    st.stop()
    
status.text('Data retrieved successfully')

window_start = ref_time - half_span
window_end = ref_time + half_span

# Raw signal plot
st.subheader('Raw Signal')
analysis_data = deepcopy(raw_signal)
raw_data = analysis_data.crop(window_start, window_end)
raw_plot = create_timeseries_plot(raw_data, 'Raw Signal Data')
st.plotly_chart(raw_plot, use_container_width=True)

# Processed signal plot
st.subheader('Processed Signal')

if noise_reduction:
    clean_signal = analysis_data.whiten()
    filtered_signal = clean_signal.bandpass(frequency_bounds[0], frequency_bounds[1])
else:
    filtered_signal = analysis_data.bandpass(frequency_bounds[0], frequency_bounds[1])

trimmed_signal = filtered_signal.crop(window_start, window_end)
processed_plot = create_timeseries_plot(trimmed_signal, 'Processed Signal Data')
st.plotly_chart(processed_plot, use_container_width=True)

# Export functionality
export_data = {'Time': trimmed_signal.times.value, 'Amplitude': trimmed_signal.value}
export_df = pd.DataFrame(export_data)
csv_content = export_df.to_csv(index=False)
encoded_data = base64.b64encode(csv_content.encode()).decode()
filename = f'{selected_observatory}-DATA-{int(window_start)}-{int(window_end-window_start)}.csv'
download_link = f'<a href="data:file/csv;base64,{encoded_data}" download="{filename}">Export Data as CSV</a>'
st.markdown(download_link, unsafe_allow_html=True)

# Audio playback
st.audio(generate_audio_stream(trimmed_signal), format='audio/wav')

with st.expander("Analysis Notes"):
    st.markdown("""
 * Noise reduction equalizes the signal across frequency bands for clearer analysis
 * Frequency filtering isolates the signal components within specified frequency bounds
""")

# Time-frequency analysis using matplotlib
st.subheader('Time-Frequency Analysis')

transformed_signal = analysis_data.q_transform(outseg=(ref_time-half_span, ref_time+half_span), qrange=resolution_range)

with render_lock:
    # Set dark background for matplotlib plot
    plt.style.use('dark_background')
    transform_plot = transformed_signal.plot()
    plot_axis = transform_plot.gca()
    transform_plot.colorbar(label="Signal Intensity", vmax=intensity_cap, vmin=0)
    plot_axis.grid(False)
    plot_axis.set_yscale('log')
    plot_axis.set_ylim(bottom=15)
    st.pyplot(transform_plot, clear_figure=True)
    plt.style.use('default')  # Reset style

with st.expander("Interpretation Guide"):
    st.markdown("""
The time-frequency visualization reveals how signal frequencies evolve over time.

 * Horizontal axis represents time progression
 * Vertical axis shows frequency components

Color intensity indicates the signal strength at each point.

The resolution factor controls the analysis precision:
 * Lower values (5-20) are optimal for rapid frequency changes
 * Higher values (80-120) better reveal gradual frequency evolution

This analysis helps identify and characterize different types of astronomical events.
""")