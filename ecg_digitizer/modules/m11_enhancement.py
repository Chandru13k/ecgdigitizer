from scipy.signal import butter, filtfilt, iirnotch

def enhance_signal(signal, config):
    """
    Applies filtering to the calibrated 1D ECG signal:
    1. 50Hz notch filter to remove powerline interference.
    2. Butterworth low-pass filter (40Hz cutoff) to remove high-frequency noise.
    """
    fs = config['sampling_rate'] # e.g., 250 Hz
    
    # 1. 50Hz Notch Filter
    f0 = config['notch_freq']  # frequency to be removed (50Hz or 60Hz)
    Q = 30.0                   # Quality factor
    b_notch, a_notch = iirnotch(f0, Q, fs)
    notched = filtfilt(b_notch, a_notch, signal)
    
    # 2. Butterworth Low-Pass Filter
    cutoff = config['lowpass_cutoff']
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b_lp, a_lp = butter(4, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b_lp, a_lp, notched)
    
    return filtered
