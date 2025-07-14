import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout, LeakyReLU, concatenate, Cropping1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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

def normalize_pulses(pulses, n_baseline=100, trap_rise=300, trap_flat=200, return_amp=False):
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
    filt_pulses = np.array([
        trapezoidal_filter(pulse, trap_rise, trap_flat)
        for pulse in pulses_bs
    ])
    # Ignore first few samples to avoid edge effects
    start = trap_rise + trap_flat
    amplitudes = np.max(np.abs(filt_pulses[:, start:]), axis=1)
    amplitudes = np.where(amplitudes == 0, 1, amplitudes)  # avoid division by zero

    # Step 3: Normalize by amplitude
    pulses_norm = pulses_bs / amplitudes[:, None]

    if return_amp: return pulses_norm, amplitudes
    else: return pulses_norm

def normalize_pulse1d(pulse, n_baseline=100, trap_rise=300, trap_flat=200, return_amp=False):
    """
    Normalize a single pulse by:
    1. Subtracting the mean of the first n_baseline samples (baseline subtraction)
    2. Calculating amplitude via trapezoidal filter and normalizing by amplitude

    Parameters:
        pulse: np.ndarray, shape (pulse_length,)
        n_baseline: int, number of samples for baseline calculation
        trap_rise: int, trapezoidal filter rise time (in samples)
        trap_flat: int, trapezoidal filter flat top (in samples)
        return_amp: bool, whether to return amplitude

    Returns:
        pulse_norm: np.ndarray, normalized pulse
        amplitude: float, (if return_amp=True)
    """
    # Step 1: Baseline subtraction
    baseline = np.mean(pulse[:n_baseline])
    pulse_bs = pulse - baseline

    # Step 2: Trapezoidal filter amplitude calculation
    filt_pulse = trapezoidal_filter(pulse_bs, trap_rise, trap_flat)
    start = trap_rise + trap_flat
    amplitude = np.max(np.abs(filt_pulse[start:]))
    if amplitude == 0:
        amplitude = 1  # avoid division by zero

    # Step 3: Normalize by amplitude
    pulse_norm = pulse_bs / amplitude

    if return_amp:
        return pulse_norm, amplitude
    else:
        return pulse_norm

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


# def build_recon_autoencoder(input_shape=(924, 1), lr=1e-4):
#     inputs = Input(shape=input_shape)
#     x = Conv1D(64, 16, activation='relu', padding='same')(inputs)
#     x = MaxPooling1D(2, padding='same')(x)      # 924 -> 462
#     x = Conv1D(32, 8, activation='relu', padding='same')(x)
#     x = MaxPooling1D(2, padding='same')(x)      # 462 -> 231
#     x = Conv1D(16, 4, activation='relu', padding='same')(x)
#     encoded = MaxPooling1D(2, padding='same')(x) # 231 -> 116

#     x = Conv1D(16, 4, activation='relu', padding='same')(encoded)
#     x = UpSampling1D(2)(x)                      # 116 -> 232
#     x = Conv1D(32, 8, activation='relu', padding='same')(x)
#     x = UpSampling1D(2)(x)                      # 232 -> 464
#     x = Conv1D(64, 16, activation='relu', padding='same')(x)
#     x = UpSampling1D(2)(x)                      # 464 -> 928

#     # Crop 2 from start and 2 from end: 928 -> 924
#     x = Cropping1D((2, 2))(x)

#     decoded = Conv1D(1, 3, activation='linear', padding='same')(x)

#     autoencoder = Model(inputs, decoded)
#     autoencoder.compile(optimizer=Adam(lr), loss='mae')
#     return autoencoder


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

def generate_flat_top_pulse(time_vector, baseline, amplitude, t_start, rise_steepness, flat_duration, fall_steepness=None):
    """Generates a flat-top pulse: fast rise, flat region, optional fall."""
    # Fast rise
    rise = 1 / (1 + np.exp(-rise_steepness * (time_vector - t_start)))
    # Flat region: turn off the rise after t_start + flat_duration
    flat = (time_vector > t_start) & (time_vector < t_start + flat_duration)
    # Optional: add a falling edge
    if fall_steepness is not None:
        t_fall = t_start + flat_duration
        fall = 1 / (1 + np.exp(fall_steepness * (time_vector - t_fall)))
        pulse = amplitude * rise * fall
    else:
        pulse = amplitude * rise
        pulse[time_vector > (t_start + flat_duration)] = amplitude
    return baseline + pulse

def generate_wavy_noise(
    n_samples,
    white_noise_level,
    colored_noise_level,
    generate_ringing_noise=True,
    ringing_prob=0.5,
    max_ringing_bursts=3,  # NEW: max number of bursts per trace
    ringing_amp_range=(0.05, 0.25),
    ringing_freq_range=(2, 12),
    ringing_damp_range=(0.008, 0.025),
    time_vector=None
):
    """Generates noise: white + colored + optional multiple decaying oscillatory (ringing) noises at random locations."""
    white_noise = np.random.normal(0, white_noise_level, n_samples)
    colored_noise_raw = np.random.normal(0, 1, n_samples)
    colored_noise_integrated = np.cumsum(colored_noise_raw)
    std_dev = np.std(colored_noise_integrated)
    if std_dev > 0:
        colored_noise_normalized = (colored_noise_integrated - np.mean(colored_noise_integrated)) / std_dev
        colored_noise = colored_noise_normalized * colored_noise_level
    else:
        colored_noise = np.zeros(n_samples)

    ringing_noise = np.zeros(n_samples)
    if generate_ringing_noise and (time_vector is not None) and (np.random.rand() < ringing_prob):
        n_bursts = np.random.randint(1, max_ringing_bursts+1)
        for _ in range(n_bursts):
            amp = np.random.uniform(*ringing_amp_range)
            freq = np.random.uniform(*ringing_freq_range)
            phase = np.random.uniform(0, 2*np.pi)
            damping = np.random.uniform(*ringing_damp_range)
            t0 = np.random.randint(0, n_samples)
            ringing = amp * np.exp(-damping * (time_vector - time_vector[t0])) \
                      * np.sin(2 * np.pi * freq * (time_vector - time_vector[t0]) + phase)
            ringing[:t0] = 0  # No ringing before t0
            ringing_noise += ringing

    return white_noise + colored_noise + ringing_noise

def simulate_pulses(
    N_PULSES=5000,
    N_SAMPLES = 4000,
    T_MAX = 2.6,
    PROB_NORMAL = 0.45,
    PROB_DOUBLE_STEP = 0.20,
    PROB_SLOW_RISER = 0.20,
    PROB_FLAT_TOP = 0.15,
    WHITE_NOISE_LEVEL = 0.03,
    COLORED_NOISE_LEVEL = 0.025,
    RINGING_NOISE=False,
    RINGING_PROB=0.7, RINGING_AMP_RANGE=(0.05, 0.25), RINGING_FREQ_RANGE=(2, 12), RINGING_DAMP_RANGE=(0.008, 0.025)
):

    """
    return two pulses, clean and noisy. npdtype array
    total of 3 type of pulses: normal, double_step, slow_riser
    """
    time_vector = np.linspace(0, T_MAX, N_SAMPLES)
    clean_pulses = np.zeros((N_PULSES, N_SAMPLES))
    noisy_pulses = np.zeros((N_PULSES, N_SAMPLES))
    
    print(f"Generating {N_PULSES} pulses with all 3 types, scaled to the target image...")
    
    for i in range(N_PULSES):
        baseline = np.random.uniform(0, 0.25)
        decay_tau = np.random.uniform(20, 30)
        
        pulse_type = np.random.choice(['normal', 'double_step', 'slow_riser', 'flat_top'],p=[PROB_NORMAL, PROB_DOUBLE_STEP, PROB_SLOW_RISER, PROB_FLAT_TOP])
    
        if pulse_type == 'slow_riser':
            # --- Modified: Use a double-sigmoid for a slow initial rise, sharper main rise ---
            # Slow initial rise
            t_start1 = np.random.uniform(0.85, 1.05)
            # amp1 = np.random.uniform(0.15, 0.22)
            steepness1 = np.random.uniform(5, 10)
            # Main fast rise
            t_start2 = t_start1 + np.random.uniform(0.12, 0.17)
            # amp2 = np.random.uniform(1.15, 1.3)
            while True:
                amp1 = np.random.uniform(0.005, 0.2)
                amp2 = np.random.uniform(0.01, 1.0 - amp1)
                if amp1 + amp2 >= 0.015:
                    break
            steepness2 = np.random.uniform(30, 50)
            clean_pulse = generate_double_step_pulse(
                time_vector, baseline,
                amp1, t_start1, steepness1,
                amp2, t_start2, steepness2,
                decay_tau
            )
    
        elif pulse_type == 'double_step':
            # --- Generate a DOUBLE-STEP pulse, correctly scaled ---
            t_start1 = np.random.uniform(0.8, 1.0)
            # amp1 = np.random.uniform(0.3, 0.5)
            # amp2 = np.random.uniform(1.0, 1.2)
            while True:
                amp1 = np.random.uniform(0.005, 0.5)
                amp2 = np.random.uniform(0.01, 1.0 - amp1)
                if amp1 + amp2 >= 0.015:
                    break
            steepness1 = np.random.uniform(20, 50)
            t_start2 = t_start1 + np.random.uniform(0.1, 0.3)
            steepness2 = np.random.uniform(60, 120)
            clean_pulse = generate_double_step_pulse(time_vector, baseline, amp1, t_start1, steepness1, amp2, t_start2, steepness2, decay_tau)
        
        elif pulse_type == 'flat_top':
            amplitude = np.random.uniform(0.8, 1.0)
            t_start = np.random.uniform(1.0, 1.1)
            rise_steepness = np.random.uniform(8, 18)
            flat_duration = np.random.uniform(0.3, 0.7)
            # Optionally, add a falling edge:
            # fall_steepness = np.random.uniform(20, 50)
            # clean_pulse = generate_flat_top_pulse(time_vector, baseline, amplitude, t_start, rise_steepness, flat_duration, fall_steepness)
            clean_pulse = generate_flat_top_pulse(time_vector, baseline, amplitude, t_start, rise_steepness, flat_duration)
    
        else: # pulse_type == 'normal'
            # amplitude = np.random.uniform(1.3, 1.5)
            amplitude = np.random.uniform(0.015, 1.0)
            t_start = np.random.uniform(1.0, 1.1)
            # steepness = np.random.uniform(70, 150)
            steepness = np.random.uniform(8, 18)
            clean_pulse = generate_single_step_pulse(time_vector, baseline, amplitude, t_start, steepness, decay_tau)
            
        noise = generate_wavy_noise(
            N_SAMPLES, WHITE_NOISE_LEVEL, COLORED_NOISE_LEVEL, generate_ringing_noise=RINGING_NOISE,
            ringing_prob=RINGING_PROB,
            ringing_amp_range=RINGING_AMP_RANGE,
            ringing_freq_range=RINGING_FREQ_RANGE,
            ringing_damp_range=RINGING_DAMP_RANGE,
            time_vector=time_vector
        )
        
        clean_pulses[i, :] = clean_pulse
        noisy_pulses[i, :] = clean_pulse + noise
    print("Generation complete.")
    return clean_pulses, noisy_pulses