import numpy as np

from scipy.signal import savgol_filter, find_peaks

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
        Calculate the rise time of the pulse (time from low_frac to high_frac of the amplitude).

        :param low_frac: Fraction for low threshold (default: 0.1)
        :param high_frac: Fraction for high threshold (default: 0.9)
        :param use_savgol: Apply Savitzky-Golay smoothing (default: True)
        :param window_length: S-G window length (default: 51)
        :param polyorder: S-G polynomial order (default: 3)
        :return: (rise_time, t_low, t_high)
        """

        # Smooth the data if requested
        if use_savgol:
            pulse_smooth = savgol_filter(self.data, window_length=window_length, polyorder=polyorder)
        else:
            pulse_smooth = self.data

        # Baseline: median of first 100 samples
        baseline = np.median(pulse_smooth[:100])

        # Detect pulse polarity and peak/trough
        max_val = np.max(pulse_smooth)
        min_val = np.min(pulse_smooth)

        # Decide if pulse is positive or negative going
        if abs(max_val - baseline) > abs(min_val - baseline):
            # Positive pulse
            peak = max_val
            low_level = baseline + low_frac * (peak - baseline)
            high_level = baseline + high_frac * (peak - baseline)
            # Find crossings after baseline
            crossing_indices = np.where(pulse_smooth > low_level)[0]
            if len(crossing_indices) == 0:
                raise ValueError("No low threshold crossing found (positive pulse).")
            idx_low = crossing_indices[0]
            crossing_indices = np.where(pulse_smooth > high_level)[0]
            idx_high = crossing_indices[crossing_indices > idx_low][0]  # after idx_low
        else:
            # Negative pulse
            peak = min_val
            low_level = baseline + low_frac * (peak - baseline)
            high_level = baseline + high_frac * (peak - baseline)
            crossing_indices = np.where(pulse_smooth < low_level)[0]
            if len(crossing_indices) == 0:
                raise ValueError("No low threshold crossing found (negative pulse).")
            idx_low = crossing_indices[0]
            crossing_indices = np.where(pulse_smooth < high_level)[0]
            idx_high = crossing_indices[crossing_indices > idx_low][0]  # after idx_low

        t_low = self.time[idx_low]
        t_high = self.time[idx_high]
        t_rise = t_high - t_low

        return t_rise, t_low, t_high

    def find_decay_time(self, low_frac: float = 0.1, high_frac: float = 0.9, use_savgol: bool = True,
                        window_length: int = 51, polyorder: int = 3):
        """
        Find decay time between high_frac and low_frac thresholds.
        :param low_frac: Lower threshold fraction (default: 0.1)
        :param high_frac: Upper threshold fraction (default: 0.9)
        :param use_savgol: Whether to apply Savitzky-Golay filter (default: True)
        :param window_length: Window length for Savitzky-Golay filter
        :param polyorder: Polynomial order for Savitzky-Golay filter
        """
        # Optionally smooth data
        if use_savgol:
            data = savgol_filter(self.data, window_length=window_length, polyorder=polyorder)
        else:
            data = self.data

        # Handle positive or negative pulses
        if np.abs(data.max()) > np.abs(data.min()):
            max_val = data.max()
            peak_idx = data.argmax()
            high = high_frac * max_val
            low = low_frac * max_val
            def crossed(x, th): return x <= th
        else:
            max_val = data.min()
            peak_idx = data.argmin()
            high = high_frac * max_val
            low = low_frac * max_val
            def crossed(x, th): return x >= th
        
        # Find first index after peak where pulse crosses high threshold
        idx_high_candidates = np.where(crossed(data[peak_idx:], high))[0]
        if idx_high_candidates.size == 0:
            return np.nan, np.nan, np.nan
        idx_high = idx_high_candidates[0] + peak_idx
        
        # Find first index after idx_high where pulse crosses low threshold
        idx_low_candidates = np.where(crossed(data[idx_high:], low))[0]
        if idx_low_candidates.size == 0:
            return np.nan, self.time[idx_high], np.nan
        idx_low = idx_low_candidates[0] + idx_high
        
        t_decay = self.time[idx_low] - self.time[idx_high]
        return t_decay, self.time[idx_high], self.time[idx_low]
    
    def find_peak_fwhm_time(self, 
        L: float = 0.9, R: float = 0.9, type: str = 'peak', peak_deriv: bool = True, return_norm: bool = False, time_space: str = 'deriv',
        use_savgol: bool = True, window_length=51, polyorder=3, height=0.4, prominence=0.43):
        """
        Calculate the peak/FWHM time of a pulse [90%/50% from each side of the peak (in derivative space for sigmoid-like pulse)]

        :param L: Lower threshold fraction (default: 0.9)
        :param R: Upper threshold fraction (default: 0.9)
        :param type: Type of time, peak time ('peak') or FWHM time ('fwhm') (default: 'peak')
        :param peak_deriv: Find peaks in the first derivative of the pulse (default: True)
        :param return_norm: Return normalized pulse; either original or derivative dependining on peak_deriv (default: False)
        :param use_savgol: Apply Savitzky-Golay smoothing (default: True)
        :param window_length: Smoothing window length (default: 51)
        :param polyorder: Smoothing polynomial order (default: 3)
        :param height: Minimum height to find number of peaks (default: 0.4)
        :param prominence: Minimum prominence to find number of peaks (default: 0.43)
        :param time_space: 'deriv' (default) for threshold crossings in derivative space for finding peak/FWHM time, 'pulse' for original pulse space.
        """

        # --- Input checks ---
        if not (isinstance(self.data, np.ndarray) and isinstance(self.time, np.ndarray)):
            raise TypeError("pulse and time must be numpy arrays.")
        if self.data.ndim != 1 or self.time.ndim != 1:
            raise ValueError("pulse and time must be 1D arrays.")
        if self.data.shape[0] != self.time.shape[0]:
            raise ValueError("pulse and time must have the same length.")

        pulse = self.data
        t = self.time
        if type not in ('peak', 'fwhm'):
            raise TypeError("type must be 'peak' or 'fwhm'.")

        if type == 'fwhm':
            L = 0.5
            R = 0.5

        # --- Derivative or pulse processing ---
        if peak_deriv:
            if use_savgol:
                derivative = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
            else:
                derivative = np.gradient(pulse)
        else:
            if use_savgol:
                derivative = savgol_filter(pulse, window_length=window_length, polyorder=polyorder)
            else:
                derivative = pulse

        max_abs = np.max(np.abs(derivative))
        if max_abs == 0:
            deriv_norm = derivative * 0
        else:
            deriv_norm = derivative / max_abs

        # --- Peak finding in normalized derivative ---
        peaks, _ = find_peaks(np.abs(deriv_norm), height=height, prominence=prominence)
        if len(peaks) == 0:
            print('no peaks found')
            return None, None, None

        # Use sign of largest peak in normalized derivative
        largest_peak_idx = peaks[np.argmax(np.abs(deriv_norm[peaks]))]
        sign = np.sign(deriv_norm[largest_peak_idx]) if deriv_norm[largest_peak_idx] != 0 else 1

        # --- Find threshold crossings in derivative space ---
        L_val = L * sign
        first_peak_idx = peaks[0]
        left = deriv_norm[:first_peak_idx+1]
        left_cross = np.where(sign * left >= sign * L_val)[0]
        if len(left_cross) == 0:
            idx_L = 0
        else:
            idx_L = left_cross[0]

        R_val = R * sign
        last_peak_idx = peaks[-1]
        right = deriv_norm[last_peak_idx:]
        right_cross = np.where(sign * right <= sign * R_val)[0]
        if len(right_cross) == 0:
            idx_R = len(t) - 1 - last_peak_idx
        else:
            idx_R = right_cross[0]

        # Interpolation for more accurate crossing
        def interp_time(idx, arr, val, t_arr, offset=0):
            if idx == 0:
                return t_arr[offset + idx]
            x0, x1 = t_arr[offset + idx - 1], t_arr[offset + idx]
            y0, y1 = arr[idx - 1], arr[idx]
            if y1 == y0:  # avoid division by zero
                return x0
            return x0 + (val - y0) * (x1 - x0) / (y1 - y0)

        # Get times in derivative space
        t_L_deriv = interp_time(idx_L, left, L_val, t)
        t_R_deriv = interp_time(idx_R, right, R_val, t, offset=last_peak_idx)

        # --- Optionally, get times in pulse space ---
        if time_space == 'pulse':
            # Normalize pulse to [0, 1] for thresholding
            pulse_norm = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse))
            # Left crossing
            left_pulse = pulse_norm[:first_peak_idx+1]
            left_cross_pulse = np.where(sign * left_pulse >= sign * L)[0]
            if len(left_cross_pulse) == 0:
                idx_L_pulse = 0
            else:
                idx_L_pulse = left_cross_pulse[0]
            t_L = interp_time(idx_L_pulse, left_pulse, L, t)
            # Right crossing
            right_pulse = pulse_norm[last_peak_idx:]
            right_cross_pulse = np.where(sign * right_pulse <= sign * R)[0]
            if len(right_cross_pulse) == 0:
                idx_R_pulse = len(t) - 1 - last_peak_idx
            else:
                idx_R_pulse = right_cross_pulse[0]
            t_R = interp_time(idx_R_pulse, right_pulse, R, t, offset=last_peak_idx)
        else:
            t_L = t_L_deriv
            t_R = t_R_deriv

        interval = t_R - t_L
        if t_L >= t_R or interval <= 0:
            # Mark as invalid
            if return_norm:
                return np.nan, np.nan, np.nan, deriv_norm
            else:
                return np.nan, np.nan, np.nan
        if return_norm:
            return abs(interval), t_L, t_R, deriv_norm
        else:
            return abs(interval), t_L, t_R

    def calculate_area(self):
        return np.sum(self.data)
    
    def trapezoidal_filter(self, data, rise: float = 120, flat: float = 400):
        """
        :param rise: rising length in unit of sampels (default: 120)
        :param flat: flat-top length in unit of samples (default: 400)
        """
        kernel = np.concatenate([
            np.ones(rise),
            np.zeros(flat),
            -np.ones(rise)
        ])
        kernel = kernel / rise
        # Apply filter
        filt = np.convolve(data, kernel, mode='same')
        return filt
    
    def normalize_pulse(self, bl: float = 100, trap_rise: float = 120, trap_flat: float = 400, return_amp: bool = False):
        """
        :param bl: baseline (default: 100)
        :param trap_rise: rising length in unit of sampels (default: 120)
        :param trap_flat: flat-top length in unit of samples (default: 400)
        """
        # Baseline subtraction
        pulse = self.data
        baseline = np.mean(pulse[:bl])
        pulse_bs = pulse - baseline

        # Trapezoidal filter amplitude calculation
        filt_pulse = self.trapezoidal_filter(pulse_bs, rise=trap_rise, flat=trap_flat)
        start = trap_rise + trap_flat
        amplitude = np.max(np.abs(filt_pulse[start:]))
        if amplitude == 0:
            amplitude = 1  # avoid division by zero

        # Normalize by amplitude
        pulse_norm = pulse_bs / amplitude
        if return_amp:
            return pulse_norm, amplitude
        else:
            return pulse_norm

    def count_peaks(self, use_savgol: bool = True, deriv: bool = True, 
                    window_length=51, polyorder=3, height=0.4, prominence=0.43):
        if deriv:
            if use_savgol:
                pulse = savgol_filter(self.data, window_length=window_length, polyorder=polyorder, deriv=1)
            else:
                pulse = np.gradient(self.data)
        else:
            if use_savgol:
                pulse = savgol_filter(self.data, window_length=window_length, polyorder=polyorder)
            else:
                pulse = self.data

        max_abs = np.max(np.abs(pulse))
        if max_abs == 0:
            deriv_norm = pulse * 0
        else:
            deriv_norm = pulse / max_abs

        peaks, _ = find_peaks(np.abs(deriv_norm), height=height, prominence=prominence)
        return len(peaks)
    
    def l1_norm(self, reference_pulse):
        """
        :param reference_pulse: an object that contains pulse (.data) and time (.time)
        """
        # Normalize both pulses
        pulse_norm = self.normalize_pulse()
        ref_norm = reference_pulse.normalize_pulse()

        t = self.time

        if len(pulse_norm) != len(ref_norm):
            raise ValueError("Pulses must have the same length.")

        # Find rising edge times on normalized pulses
        _, t_low_pulse, _ = self.find_rise_time()
        _, t_low_ref, _ = reference_pulse.find_rise_time()

        # Calculate shift (in samples)
        dt = t[1] - t[0]
        shift_samples = int(round((t_low_pulse - t_low_ref) / dt))

        # Shift reference_pulse to align with self
        if shift_samples > 0:
            ref_aligned = np.pad(ref_norm, (shift_samples, 0), mode='constant')[:len(pulse_norm)]
            pulse_aligned = pulse_norm
        elif shift_samples < 0:
            pulse_aligned = np.pad(pulse_norm, (-shift_samples, 0), mode='constant')[:len(ref_norm)]
            ref_aligned = ref_norm
        else:
            pulse_aligned = pulse_norm
            ref_aligned = ref_norm

        # Truncate to common length
        min_len = min(len(pulse_aligned), len(ref_aligned))
        pulse_aligned = pulse_aligned[:min_len]
        ref_aligned = ref_aligned[:min_len]

        # Compute L1 norm
        l1_norm = np.sum(np.abs(pulse_aligned - ref_aligned))
        return l1_norm

    def find_amplitude(self, n_baseline: float = 100, rise: float = 120, flat: float =400):
        baseline = np.mean(self.data[:n_baseline])
        pulse_bs = self.data - baseline
        filt = self.trapezoidal_filter(pulse_bs, rise, flat)
        # Ignore edge effects
        start = rise + flat
        amplitude = np.max(np.abs(filt[start:]))
        if amplitude == 0:
            amplitude = 1
        return amplitude

    def normalize_deriv(self, window_length: int = 51, polyorder: int = 3):
        """
        Input original pulse and return a normalized derivative of a 1D pulse.
        :param: window_length: window length for savgol filter
        :param: polyorder: polyorder for savgol filter
        """
        pulse=self.data
        # Smooth and differentiate
        deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
        
        # Find main peak
        peak_idx = np.argmax(np.abs(deriv))
        peak_val = deriv[peak_idx]
        
        # Avoid division by zero
        if peak_val == 0:
            peak_val = 1
        
        # Normalize so main peak is +1 or -1
        deriv_norm = deriv / peak_val
        
        return deriv_norm