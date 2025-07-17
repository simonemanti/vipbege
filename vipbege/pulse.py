import numpy as np

from scipy.signal import savgol_filter

class Pulse:
    def __init__(self, data: np.ndarray, sampling_rate: float = 400e6, label: str = None):
        """
        :param data: 1D numpy array representing the pulse data
        :param sampling_rate: Sampling rate in Hz (default: 400 MHz)
        :param label: Optional label for the pulse
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.label = label
        
        # Calculate time axis in microseconds
        n_channels = len(data)
        dt = 1 / sampling_rate
        self.time = np.arange(n_channels) * dt * 1e6

    def find_rise_time(self, low_frac: float = 0.1, high_frac: float = 0.9, 
                       use_savgol: bool = True, window_length: int = 51, polyorder: int = 3):
        """
        :param low_frac: Lower threshold fraction (default: 0.1)
        :param high_frac: Upper threshold fraction (default: 0.9)
        :param use_savgol: Apply Savitzky-Golay smoothing (default: True)
        :param window_length: Smoothing window length (default: 51)
        :param polyorder: Smoothing polynomial order (default: 3)
        """
        if use_savgol:
            pulse_smooth = savgol_filter(self.data, window_length=window_length, polyorder=polyorder)
        else:
            pulse_smooth = self.data
        
        baseline = np.median(pulse_smooth[:100])
        plateau = np.median(pulse_smooth[-100:])
        
        if plateau > baseline:
            low_level = baseline + low_frac * (plateau - baseline)
            high_level = baseline + high_frac * (plateau - baseline)
            idx_low = np.where(pulse_smooth >= low_level)[0]
            idx_high = np.where(pulse_smooth >= high_level)[0]
        else:
            low_level = baseline + (1 - low_frac) * (plateau - baseline)
            high_level = baseline + (1 - high_frac) * (plateau - baseline)
            idx_low = np.where(pulse_smooth <= low_level)[0]
            idx_high = np.where(pulse_smooth <= high_level)[0]
            
        if len(idx_low) == 0 or len(idx_high) == 0:
            raise ValueError("Could not find threshold crossings in pulse.")
            
        idx_low = idx_low[0]
        idx_high = idx_high[0]
        t_rise = self.time[idx_high] - self.time[idx_low]
        
        return t_rise, self.time[idx_low], self.time[idx_high]
