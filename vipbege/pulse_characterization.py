import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, argrelmin
from scipy.signal import find_peaks


def find_rise_time(t, pulse, low_frac=0.1, high_frac=0.9, window_length=51, polyorder=3):
    pulse_smooth = savgol_filter(pulse, window_length=window_length, polyorder=polyorder)

    # Step: from baseline (start) to plateau (end)
    baseline = np.median(pulse_smooth[:100])      # first 100 samples
    plateau  = np.median(pulse_smooth[-100:])     # last 100 samples

    # Rising step: plateau > baseline, Falling step: plateau < baseline
    if plateau > baseline:
        # Rising step (like your plot)
        low_level  = baseline + low_frac  * (plateau - baseline)
        high_level = baseline + high_frac * (plateau - baseline)
        idx_low = np.where(pulse_smooth >= low_level)[0]
        idx_high = np.where(pulse_smooth >= high_level)[0]
    else:
        # Falling step
        low_level  = baseline + (1 - low_frac)  * (plateau - baseline)
        high_level = baseline + (1 - high_frac) * (plateau - baseline)
        idx_low = np.where(pulse_smooth <= low_level)[0]
        idx_high = np.where(pulse_smooth <= high_level)[0]

    if len(idx_low) == 0 or len(idx_high) == 0:
        raise ValueError("Could not find threshold crossings in pulse.")

    # Take the first crossing
    idx_low = idx_low[0]
    idx_high = idx_high[0]

    t_rise = t[idx_high] - t[idx_low]
    return t_rise, t[idx_low], t[idx_high]

# only works for positve pulses
# def find_rise_time(t, pulse, low_frac=0.1, high_frac=0.9, window_length=100, polyorder=3):
#     pulse_smooth = savgol_filter(pulse, window_length=window_length, polyorder=polyorder)
    
#     max_val = pulse_smooth.max()
#     low = low_frac * max_val
#     high = high_frac * max_val

#     idx_low = np.where(pulse_smooth >= low)[0][0]
#     idx_high = np.where(pulse_smooth >= high)[0][0]
    
#     t_rise = t[idx_high] - t[idx_low]
#     return t_rise, t[idx_low], t[idx_high]


def find_decay_time(t, pulse, low_frac=0.1, high_frac=0.9):

    max_val = pulse.max()
    peak_idx = pulse.argmax()
    high = high_frac * max_val
    low = low_frac * max_val

    # Find index after the peak where pulse falls below high
    idx_high_candidates = np.where(pulse[peak_idx:] <= high)[0]
    if idx_high_candidates.size == 0:
        return np.nan, np.nan, np.nan  # Could not find high threshold crossing after peak
    idx_high = idx_high_candidates[0] + peak_idx

    idx_low_candidates = np.where(pulse[idx_high:] <= low)[0]
    if idx_low_candidates.size == 0:
        return np.nan, t[idx_high], np.nan
    idx_low = idx_low_candidates[0] + idx_high

    t_decay = t[idx_low] - t[idx_high]
    return t_decay, t[idx_high], t[idx_low]


def find_peak_time(t, pulse, low_frac=0.1, high_frac=0.9):
    tL = find_rise_time(t,pulse, low_frac, high_frac)[2]
    tR = find_decay_time(t,pulse, low_frac, high_frac)[1]
    if np.isnan(tL) or np.isnan(tR):
        return np.nan
    else:
        return abs(tL - tR), tL, tR


def area(pulse):
    return np.sum(pulse)


def within_std(x, value, nbins=400, return_value=False):
    if np.isnan(value):
        within_bounds = np.nan
    else:
        counts, bin_edges = np.histogram(x, bins=nbins)
        max_bin_index = np.argmax(counts)
        max_bin_center = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        std = np.std(x)

        lower_bound = max_bin_center - std
        upper_bound = max_bin_center + std
        within_bounds = lower_bound <= value <= upper_bound

    if return_value:
        return within_bounds, value
    else:
        return within_bounds


def smooth_pulse(pulse, return_deriv=False, **kwargs):
    smoothed_pulse = savgol_filter(pulse, **kwargs)
    if return_deriv:
        return smoothed_pulse, np.gradient(smoothed_pulse)
    else:
        return smoothed_pulse
 

def get_peaks(pulse, window_length=100, polyorder=3, deriv=0, prominence=0.3, height=None):
    s = savgol_filter(pulse, window_length, polyorder, deriv)
    s = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s*0
    peaks, _ = find_peaks(s, prominence=prominence, height=height)
    return peaks, pulse[peaks]

def count_peaks_tot(pulses, **kwargs):
    peak_list = [len(get_peaks(row, **kwargs)[0]) for row in pulses]
    return np.array(peak_list)

def count_peaks_fd_sd(
    pulse,
    winlen_pulse=61, poly_pulse=2,
    winlen_fd=31, poly_fd=2,
    winlen_sd=31, poly_sd=2,
    peak_prominence=0.05, peak_height=0.1,
    get_peaks_func=get_peaks
):
    """
    Returns:
        n_peaks_fd: Number of peaks in smoothed, normalized 1st derivative of smoothed pulse
        n_peaks_sd: Number of peaks in smoothed, normalized 2nd derivative of smoothed 1st derivative
    """
    if get_peaks_func is None:
        raise ValueError("You must provide a get_peaks function.")

    # Smooth the original pulse
    pulse_smoothed = savgol_filter(pulse, window_length=winlen_pulse, polyorder=poly_pulse)

    # 1st derivative of smoothed pulse
    first_deriv = np.gradient(pulse_smoothed)
    # Smooth and normalize
    first_deriv_smooth = savgol_filter(first_deriv, window_length=winlen_fd, polyorder=poly_fd)
    max_abs_fd = np.max(np.abs(first_deriv_smooth))
    first_deriv_smooth_norm = first_deriv_smooth / max_abs_fd if max_abs_fd != 0 else first_deriv_smooth

    # Find peaks in 1st derivative
    peaks_fd, _ = get_peaks_func(first_deriv_smooth_norm, prominence=peak_prominence, height=peak_height)

    # 2nd derivative of smoothed 1st derivative
    second_deriv = np.gradient(first_deriv_smooth)
    # Smooth and normalize
    second_deriv_smooth = savgol_filter(second_deriv, window_length=winlen_sd, polyorder=poly_sd)
    max_abs_sd = np.max(np.abs(second_deriv_smooth))
    second_deriv_smooth_norm = second_deriv_smooth / max_abs_sd if max_abs_sd != 0 else second_deriv_smooth

    # Find peaks in 2nd derivative
    peaks_sd, _ = get_peaks_func(second_deriv_smooth_norm, prominence=peak_prominence, height=peak_height)

    return len(peaks_fd), len(peaks_sd)


def mean_pulse(pulses):
    return np.mean(pulses, axis=0)

def l1_norm_to_mean(pulse, reference_pulse, t=None):
    """
    Aligns rising edge of pulse and reference_pulse, then computes L1 norm.
    t: time array (same length as pulse), if None, assume np.arange(len(pulse))
    """
    if t is None:
        t = np.arange(len(pulse))

    # Find rising edge indices
    _, t_low_pulse, _ = find_rise_time(t, pulse)
    _, t_low_ref, _ = find_rise_time(t, reference_pulse)

    # Calculate shift (in samples)
    shift = int(round(t_low_pulse - t_low_ref))

    # Shift reference_pulse to align with pulse
    if shift > 0:
        # pulse rises later, pad reference_pulse at start
        ref_aligned = np.pad(reference_pulse, (shift, 0), mode='constant')[:len(reference_pulse)]
        pulse_aligned = pulse
    elif shift < 0:
        # pulse rises earlier, pad pulse at start
        pulse_aligned = np.pad(pulse, (-shift, 0), mode='constant')[:len(pulse)]
        ref_aligned = reference_pulse
    else:
        # No shift needed
        pulse_aligned = pulse
        ref_aligned = reference_pulse

    # Compute L1 norm between aligned pulses
    l1_norm = np.sum(np.abs(np.asarray(pulse_aligned) - np.asarray(ref_aligned)))
    return l1_norm

# def l1_norm_to_mean(pulse, reference_pulse):
#     """
#     Computes the L1 norm between each pulse and the mean pulse.

#     Parameters:
#         pulses: 2D numpy array of shape (n_samples, n_time)

#     Returns:
#         l1_norms: 1D numpy array of shape (n_samples,)
#     """
#     # Compute L1 norm for each pulse
#     l1_norms = np.sum(np.abs(np.asarray(pulse) - np.asarray(reference_pulse)))
#     return l1_norms


def peak_normalize(data):
    """
    Normalizes each pulse (row) in the dataset to the [0, 1] range.
    """
    # Find the min and max for each pulse (row)
    X_min = data.min(axis=1, keepdims=True)
    X_max = data.max(axis=1, keepdims=True)
    
    # Calculate the range, adding a small epsilon to avoid division by zero for flat lines
    pulse_range = X_max - X_min
    epsilon = 1e-10
    
    # Perform the normalization
    normalized_data = (data - X_min) / (pulse_range + epsilon)
    
    return normalized_data


# --------- below is specifically for 2021-type waveform -----------------------
def find_LR_crossings_derivative(t, pulse, L=0.9, R=0.9, window_length=100, polyorder=3):
    """
    Find the times where the Savitzky-Golay smoothed derivative of the pulse crosses
    L% and R% of its maximum value (on the rising and falling edges around the main peak).

    Parameters:
        t : array-like
            Time array (same length as pulse)
        pulse : array-like
            Original pulse signal
        L : float
            Fraction (e.g., 0.1 for 10%)
        R : float
            Fraction (e.g., 0.9 for 90%)
        window_length : int
            Window length for Savitzky-Golay filter (must be odd)
        polyorder : int
            Polynomial order for Savitzky-Golay filter

    Returns:
        peak_time : float
            Time of the max derivative
        t_L : float
            Time where derivative first crosses L% of peak (on rising edge)
        t_R : float
            Time where derivative falls to R% of peak (on falling edge)
        deriv_norm : array
            The normalized derivative (for plotting if desired)
    """
    # Compute smoothed derivative
    deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
    # Normalize
    deriv_norm = (deriv - np.min(deriv)) / (np.max(deriv) - np.min(deriv)) if np.max(deriv) != np.min(deriv) else deriv * 0

    # Find peak
    peak_idx = np.argmax(deriv_norm)
    peak_val = deriv_norm[peak_idx]
    peak_time = t[peak_idx]

    # L% and R% of peak value
    L_val = L * peak_val
    R_val = R * peak_val

    # Rising edge: from start to peak, find first crossing >= L_val
    left_idx = np.where(deriv_norm[:peak_idx] >= L_val)[0]
    t_L = t[left_idx[0]] if len(left_idx) > 0 else t[0]

    # Falling edge: from peak to end, find first crossing <= R_val
    right_idx = np.where(deriv_norm[peak_idx:] <= R_val)[0]
    t_R = t[peak_idx + right_idx[0]] if len(right_idx) > 0 else t[-1]

    return peak_time, t_L, t_R, deriv_norm

def find_amplitudes(pulse, window_length=100, polyorder=3):
    """
    Smooth each row of a 2D array and return the max of each smoothed row.
    
    Parameters:
        pulse: np.ndarray, shape (n_rows, n_cols)
        window_length: int, window size for smoothing (must be odd and <= n_cols)
        polyorder: int, polynomial order for Savitzky-Golay filter
        
    Returns:
        amplitudes: np.ndarray, shape (n_rows,)
    """
    # Smooth each row and find max
    smoothed = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, axis=1)
    amplitudes = np.max(smoothed, axis=1) - np.min(smoothed, axis=1)
    return amplitudes