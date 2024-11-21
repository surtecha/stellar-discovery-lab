from scipy import signal
from scipy.io import wavfile
import io
import numpy as np

def generate_audio_stream(signal_data, reference_time=None):
    transition_width = 1.0/10
    envelope = signal.windows.tukey(len(signal_data), alpha=transition_width)
    modulated_signal = signal_data * envelope

    amplitude_scale = 32767 * 0.9
    normalized_signal = np.int16(modulated_signal/np.max(np.abs(modulated_signal)) * amplitude_scale)

    sample_rate = 1/normalized_signal.dt.value
    audio_buffer = io.BytesIO()    
    wavfile.write(audio_buffer, int(sample_rate), normalized_signal)
    
    return audio_buffer