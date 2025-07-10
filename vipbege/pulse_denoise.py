import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout, LeakyReLU, concatenate, Cropping1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def normalize_pulses(pulses, n_baseline=100, trap_rise=100, trap_flat=20):
    """
    Normalize pulses by:
    1. Subtracting the mean of the first n_baseline samples (baseline subtraction)
    2. Calculating amplitude via trapezoidal filter and normalizing by amplitude

    Parameters:
        pulses: np.ndarray, shape (n_pulses, pulse_length)
        n_baseline: int, number of samples for baseline calculation
        trap_rise: int, trapezoidal filter rise time (in samples)
        trap_flat: int, trapezoidal filter flat top (in samples)

    Returns:
        pulses_out: np.ndarray, normalized pulses
    """
    # Step 1: Baseline subtraction
    baseline = np.mean(pulses[:, :n_baseline], axis=1, keepdims=True)
    pulses_bs = pulses - baseline

    # Step 2: Trapezoidal filter amplitude calculation
    def trap_filter(pulse):
        filt = np.zeros_like(pulse)
        for i in range(trap_rise + trap_flat, len(pulse)):
            filt[i] = (
                np.sum(pulse[i-trap_rise-trap_flat:i-trap_flat]) -
                np.sum(pulse[i-2*trap_rise-trap_flat:i-trap_rise-trap_flat])
            ) / trap_rise
        return filt

    amplitudes = np.array([
        np.max(np.abs(trap_filter(pulse)))
        for pulse in pulses_bs
    ])
    amplitudes = np.where(amplitudes == 0, 1, amplitudes)  # avoid division by zero

    # Step 3: Normalize by amplitude
    pulses_norm = pulses_bs / amplitudes[:, None]

    return pulses_norm

def normalize_pulse1d(pulse, n_baseline=100, trap_rise=100, trap_flat=20):
    """
    Normalize a single 1D pulse by:
    1. Subtracting the mean of the first n_baseline samples (baseline subtraction)
    2. Calculating amplitude via trapezoidal filter and normalizing by amplitude

    Parameters:
        pulse: np.ndarray, shape (pulse_length,)
        n_baseline: int, number of samples for baseline calculation
        trap_rise: int, trapezoidal filter rise time (in samples)
        trap_flat: int, trapezoidal filter flat top (in samples)

    Returns:
        pulse_norm: np.ndarray, normalized pulse
    """
    # Step 1: Baseline subtraction
    baseline = np.mean(pulse[:n_baseline])
    pulse_bs = pulse - baseline

    # Step 2: Trapezoidal filter amplitude calculation
    def trap_filter(pulse):
        filt = np.zeros_like(pulse)
        for i in range(trap_rise + trap_flat, len(pulse)):
            filt[i] = (
                np.sum(pulse[i-trap_rise-trap_flat:i-trap_flat]) -
                np.sum(pulse[i-2*trap_rise-trap_flat:i-trap_rise-trap_flat])
            ) / trap_rise
        return filt

    filt = trap_filter(pulse_bs)
    amplitude = np.max(np.abs(filt))
    amplitude = amplitude if amplitude != 0 else 1  # avoid division by zero

    # Step 3: Normalize by amplitude
    pulse_norm = pulse_bs / amplitude

    return pulse_norm

# def normalize_pulses(pulses):
#     amplitudes = np.max(np.abs(pulses), axis=1, keepdims=True)  # shape: (n_samples, 1)
#     return pulses / amplitudes, amplitudes

# def normalize_pulses(pulses, mode='normalize', min_val=None, max_val=None):
#     """
#     Normalize or denormalize pulses to/from [-1, 1].

#     Parameters:
#         pulses: np.ndarray, shape (n_pulses, pulse_length)
#         mode: 'normalize' or 'denormalize'
#         min_val: float or np.ndarray, minimum value(s) for normalization
#         max_val: float or np.ndarray, maximum value(s) for normalization

#     Returns:
#         If mode == 'normalize':
#             pulses_norm: normalized pulses, shape (n_pulses, pulse_length)
#             min_val: minimum value(s) used
#             max_val: maximum value(s) used
#         If mode == 'denormalize':
#             pulses_denorm: denormalized pulses, shape (n_pulses, pulse_length)
#     """
#     if mode == 'normalize':
#         # Compute min and max per pulse
#         min_val = np.min(pulses, axis=1, keepdims=True)
#         max_val = np.max(pulses, axis=1, keepdims=True)
#         # Avoid division by zero
#         denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
#         pulses_norm = 2 * (pulses - min_val) / denom - 1
#         return pulses_norm, min_val, max_val

#     elif mode == 'denormalize':
#         if min_val is None or max_val is None:
#             raise ValueError("min_val and max_val must be provided for denormalization.")
#         denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
#         pulses_denorm = (pulses + 1) / 2 * denom + min_val
#         return pulses_denorm
# 
    # else:
    #     raise ValueError("mode must be 'normalize' or 'denormalize'.")

def add_noise(pulses, noise_std=0.001, random_seed=None):
    """
    Add Gaussian noise to each pulse in a 2D array.
    pulses: 2D numpy array (n_pulses, n_samples)
    noise_std: standard deviation of the noise
    random_seed: for reproducibility
    Returns: noisy pulses (same shape as input)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=pulses.shape)
    return pulses + noise


def build_recon_autoencoder(input_shape=(924, 1), lr=1e-4):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 16, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2, padding='same')(x)      # 924 -> 462
    x = Conv1D(32, 8, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)      # 462 -> 231
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x) # 231 -> 116

    x = Conv1D(16, 4, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)                      # 116 -> 232
    x = Conv1D(32, 8, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)                      # 232 -> 464
    x = Conv1D(64, 16, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)                      # 464 -> 928

    # Crop 2 from start and 2 from end: 928 -> 924
    x = Cropping1D((2, 2))(x)

    decoded = Conv1D(1, 3, activation='linear', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(lr), loss='mae')
    return autoencoder


# -------------- pulse simulation -----------------------------
def generate_single_step_pulse(time_vector, baseline, amplitude, t_start, steepness, decay_tau):
    """Generates a standard single-stage pulse."""
    rising_edge = 1 / (1 + np.exp(-steepness * (time_vector - t_start)))
    decay_tail = np.exp(-np.maximum(0, time_vector - t_start) / decay_tau)
    return baseline + amplitude * rising_edge * decay_tail

def generate_double_step_pulse(time_vector, baseline, amp1, t_start1, steepness1, amp2, t_start2, steepness2, decay_tau):
    """Generates a pulse with a two-stage rise."""
    step1 = amp1 / (1 + np.exp(-steepness1 * (time_vector - t_start1)))
    step2 = amp2 / (1 + np.exp(-steepness2 * (time_vector - t_start2)))
    combined_rise = step1 + step2
    decay_tail = np.exp(-np.maximum(0, time_vector - t_start1) / decay_tau)
    return baseline + combined_rise * decay_tail

def generate_wavy_noise(n_samples, white_noise_level, colored_noise_level):
    """Generates the wavy noise from the target image."""
    white_noise = np.random.normal(0, white_noise_level, n_samples)
    colored_noise_raw = np.random.normal(0, 1, n_samples)
    colored_noise_integrated = np.cumsum(colored_noise_raw)
    std_dev = np.std(colored_noise_integrated)
    if std_dev > 0:
        colored_noise_normalized = (colored_noise_integrated - np.mean(colored_noise_integrated)) / std_dev
        colored_noise = colored_noise_normalized * colored_noise_level
    else:
        colored_noise = np.zeros(n_samples)
    return white_noise + colored_noise