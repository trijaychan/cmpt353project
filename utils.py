import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal, fft

def get_clean_data(data): 
    # butterworth filter
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    a_total = signal.filtfilt(b, a, data["ay"])
    
    # fast fourier transform
    yf = np.fft.fft(a_total)
    xf = np.fft.fftfreq(a_total.size)
    
    # find min value
    min_magnitude = (np.max(np.abs(yf)) / 3)
    
    # use fft to de-noise
    indices = np.abs(yf) > min_magnitude
    fhat = indices * yf
    ffilt = np.fft.ifft(fhat)
    
    return data["time"], ffilt

def get_steps_per_minute(filepath):
    data = pd.read_csv(filepath, parse_dates=['time'])
    x, y = get_clean_data(data)
    
    peaks, _ = signal.find_peaks(y)
    steps = peaks.size

    time_start = x.iloc[peaks[0]]
    time_end = x.iloc[peaks[-1]]
    
    time_in_seconds = (time_end - time_start).total_seconds()
    time_in_minutes = time_in_seconds / 60
    
    return steps / time_in_minutes

# reference: https://physics.stackexchange.com/questions/41653/how-do-i-get-the-total-acceleration-from-3-axes
def get_acceleration_from_3d(a_x, a_y, a_z):
    a_xz = np.sqrt(a_x ** 2 + a_z ** 2)
    a_total = np.sqrt(a_y ** 2 + a_xz ** 2)
    
    return a_total