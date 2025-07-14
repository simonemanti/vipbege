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


def find_decay_time(t, pulse, low_frac=0.1, high_frac=0.9):

    # Handle positive or negative pulses
    if np.abs(pulse.max()) > np.abs(pulse.min()):
        max_val = pulse.max()
        peak_idx = pulse.argmax()
        high = high_frac * max_val
        low = low_frac * max_val
        def crossed(x, th): return x <= th
    else:
        max_val = pulse.min()
        peak_idx = pulse.argmin()
        high = high_frac * max_val
        low = low_frac * max_val
        def crossed(x, th): return x >= th

    # Find first index after peak where pulse crosses high threshold
    idx_high_candidates = np.where(crossed(pulse[peak_idx:], high))[0]
    if idx_high_candidates.size == 0:
        return np.nan, np.nan, np.nan
    idx_high = idx_high_candidates[0] + peak_idx

    # Find first index after idx_high where pulse crosses low threshold
    idx_low_candidates = np.where(crossed(pulse[idx_high:], low))[0]
    if idx_low_candidates.size == 0:
        return np.nan, t[idx_high], np.nan
    idx_low = idx_low_candidates[0] + idx_high

    # (Optional) interpolate crossing times for t_high and t_low

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
 

def get_peaks(pulse, window_length=100, polyorder=3, deriv=0, prominence=0.3, height=None):
    s = savgol_filter(pulse, window_length, polyorder, deriv)
    s = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s*0
    peaks, _ = find_peaks(s, prominence=prominence, height=height)
    return peaks, pulse[peaks]

def count_peaks_tot(pulses, **kwargs):
    peak_list = [len(get_peaks(row, **kwargs)[0]) for row in pulses]
    return np.array(peak_list)


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



# --------- below is specifically for 2021-type waveform -----------------------
def find_LR_crossings_derivative(
    t, pulse, L=0.9, R=0.9, window_length=51, polyorder=3, height=0.4, prominence=0.43
):

    """
    calculate L% from the left and R% from the right of the peak/peaks
    return
    peak_times: a list of where the peaks are located
    t_L time L% from the left
    t_R time L% from the right
    deriv_norm normalized derivative from to 1, normalized by amplitude
    """
    deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
    
    max_abs = np.max(np.abs(deriv))
    if max_abs == 0:
        deriv_norm = deriv * 0
    else:
        deriv_norm = deriv / max_abs
    
    peaks, _ = find_peaks(np.abs(deriv_norm), height=height, prominence=prominence)
    
    if len(peaks) == 0:
        print('no peaks found')
        return [], None, None, deriv_norm

    peak_times = [t[p] for p in peaks]
    
    # Get sign from the largest peak in normalized derivative
    largest_peak_idx = peaks[np.argmax(np.abs(deriv_norm[peaks]))]
    sign = np.sign(deriv_norm[largest_peak_idx]) if deriv_norm[largest_peak_idx] != 0 else 1

    # t_L: L% up to the first peak (using normalized by largest peak)
    L_val = L * sign
    first_peak_idx = peaks[0]
    left = deriv_norm[:first_peak_idx+1]
    left_cross = np.where(sign * left >= sign * L_val)[0]
    if len(left_cross) == 0:
        t_L = t[0]
    else:
        idx_L = left_cross[0]
        if idx_L == 0:
            t_L = t[idx_L]
        else:
            x0, x1 = t[idx_L-1], t[idx_L]
            y0, y1 = left[idx_L-1], left[idx_L]
            t_L = x0 + (L_val - y0) * (x1 - x0) / (y1 - y0)

    # t_R: R% down from the last peak (using normalized by largest peak)
    R_val = R * sign
    last_peak_idx = peaks[-1]
    right = deriv_norm[last_peak_idx:]
    right_cross = np.where(sign * right <= sign * R_val)[0]
    if len(right_cross) == 0:
        t_R = t[-1]
    else:
        idx_R = right_cross[0]
        if idx_R == 0:
            t_R = t[last_peak_idx + idx_R]
        else:
            x0, x1 = t[last_peak_idx + idx_R - 1], t[last_peak_idx + idx_R]
            y0, y1 = right[idx_R-1], right[idx_R]
            t_R = x0 + (R_val - y0) * (x1 - x0) / (y1 - y0)

    return peak_times, t_L, t_R, deriv_norm

# def find_LR_crossings_derivative(
#     t, pulse, L=0.9, R=0.9, window_length=51, polyorder=3
# ):
#     """
#     Find the times where the Savitzky-Golay smoothed derivative of the pulse crosses
#     L% and R% of its main peak value (on the rising and falling edges around the main peak).
#     Handles both positive and negative pulses, with linear interpolation for accuracy.

#     Returns:
#         peak_time : float
#             Time of the main derivative peak
#         t_L : float
#             Time where derivative first crosses L% of main peak
#         t_R : float
#             Time where derivative first crosses R% of main peak
#         deriv_norm : array
#             The normalized derivative (for plotting if desired)
#     """
#     # Smooth and differentiate
#     deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)

#     # Find main peak (positive or negative)
#     peak_idx = np.argmax(np.abs(deriv))
#     peak_val = deriv[peak_idx]
#     sign = np.sign(peak_val) if peak_val != 0 else 1

#     # Normalize so main peak is always +1 or -1
#     deriv_norm = deriv / peak_val if peak_val != 0 else deriv * 0

#     # L/R values (on the same side as the main peak)
#     L_val = L * sign
#     R_val = R * sign

#     # Find L% crossing before peak (rising edge)
#     left = deriv_norm[:peak_idx+1]
#     left_cross = np.where(sign * left >= sign * L_val)[0]
#     if len(left_cross) == 0:
#         t_L = t[0]
#     else:
#         idx_L = left_cross[0]
#         if idx_L == 0:
#             t_L = t[idx_L]
#         else:
#             # Linear interpolation
#             x0, x1 = t[idx_L-1], t[idx_L]
#             y0, y1 = left[idx_L-1], left[idx_L]
#             t_L = x0 + (L_val - y0) * (x1 - x0) / (y1 - y0)

#     # Find R% crossing after peak (falling edge)
#     right = deriv_norm[peak_idx:]
#     right_cross = np.where(sign * right <= sign * R_val)[0]
#     if len(right_cross) == 0:
#         t_R = t[-1]
#     else:
#         idx_R = right_cross[0]
#         if idx_R == 0:
#             t_R = t[peak_idx + idx_R]
#         else:
#             # Linear interpolation
#             x0, x1 = t[peak_idx + idx_R - 1], t[peak_idx + idx_R]
#             y0, y1 = right[idx_R-1], right[idx_R]
#             t_R = x0 + (R_val - y0) * (x1 - x0) / (y1 - y0)

#     peak_time = t[peak_idx]

#     return peak_time, t_L, t_R, deriv_norm


def trapezoidal_filter(pulse, rise, flat):
    # Create trapezoidal filter kernel
    kernel = np.concatenate([
        np.ones(rise),
        np.zeros(flat),
        -np.ones(rise)
    ])
    kernel = kernel / rise
    # Apply filter
    filt = np.convolve(pulse, kernel, mode='same')
    return filt

def find_amplitudes(pulses, n_baseline=100, trap_rise=300, trap_flat=200):
    """
    Normalize pulses by:
    1. Subtracting the mean of the first n_baseline samples (baseline subtraction)
    2. Calculating amplitude via trapezoidal filter amplitude

    Parameters:
        pulses: np.ndarray, shape (n_pulses, pulse_length)
        n_baseline: int, number of samples for baseline calculation
        trap_rise: int, trapezoidal filter rise time (in samples) (current value for 2025, 120 for 2021)
        trap_flat: int, trapezoidal filter flat top (in samples) (current value for 2025, 400 for 2021)

    Returns:
        amplitudes: np.ndarray, amplitudes
    """
    # Step 1: Baseline subtraction
    baseline = np.mean(pulses[:, :n_baseline], axis=1, keepdims=True)
    pulses_bs = pulses - baseline

    # Step 2: Trapezoidal filter amplitude calculation
    filt_pulses = np.array([
        trapezoidal_filter(pulse, trap_rise, trap_flat)
        for pulse in pulses_bs
    ])
    # Ignore first few samples to avoid edge effects
    start = trap_rise + trap_flat
    amplitudes = np.max(np.abs(filt_pulses[:, start:]), axis=1)
    amplitudes = np.where(amplitudes == 0, 1, amplitudes)  # avoid division by zero

    return amplitudes

def normalize_deriv(pulse, window_length=51, polyorder=3):
    """
    Normalize derivative of pulse(s).
    If input is 2D (n_samples, n_points), returns array of same shape.
    """
    # Smooth and differentiate
    deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)

    # If 1D input, expand dims to make code below work
    if deriv.ndim == 1:
        deriv = deriv[None, :]  # shape (1, n_points)

    # Find main peak for each sample
    peak_idx = np.argmax(np.abs(deriv), axis=1)  # shape: (n_samples,)
    peak_val = np.array([deriv[i, idx] for i, idx in enumerate(peak_idx)])  # shape: (n_samples,)

    # Avoid division by zero
    peak_val[peak_val == 0] = 1

    # Normalize so main peak is always +1 or -1
    deriv_norm = deriv / peak_val[:, None]

    # If input was 1D, return 1D
    if pulse.ndim == 1:
        deriv_norm = deriv_norm[0]

    return deriv_norm