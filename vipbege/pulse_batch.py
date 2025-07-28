import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple
import struct
import os
import glob
import math
import pandas as pd


class PulseBatch:

    def __init__(self, data: np.ndarray, sampling_rate: float = 400e6, label: str = None):
        """
        :param data: 2D numpy array representing the pulse data
        :param sampling_rate: Sampling rate in Hz (default: 400 MHz)
        :param label: Optional label for the pulse
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.label = label

        if data.ndim != 2:
            raise ValueError('Inccorect input shape.')
        
        # Calculate time axis in microseconds
        n_channels = data.shape[1]
        dt = 1 / sampling_rate
        self.time = np.arange(n_channels) * dt * 1e6

    def find_rise_time(self, low_frac: float = 0.1, high_frac: float = 0.9, 
                    use_savgol: bool = True, window_length: int = 51, polyorder: int = 3):
        """
        Calculate the rise time for each pulse (time from low_frac to high_frac of the amplitude).
        :return: (t_rise, t_low, t_high) -- each is an array of shape (n_pulse,)
        """
        pulses = self.data           # shape: (n_pulse, n_samples)
        t = self.time                # shape: (n_samples,)

        n_pulse, n_samples = pulses.shape

        # Smoothing
        if use_savgol:
            pulses_smooth = np.array([
                savgol_filter(pulse, window_length=window_length, polyorder=polyorder)
                for pulse in pulses
            ])
        else:
            pulses_smooth = pulses

        baselines = np.median(pulses_smooth[:, :100], axis=1)  # shape: (n_pulse,)

        max_vals = np.max(pulses_smooth, axis=1)
        min_vals = np.min(pulses_smooth, axis=1)

        t_rise = np.full(n_pulse, np.nan)
        t_low = np.full(n_pulse, np.nan)
        t_high = np.full(n_pulse, np.nan)

        for i in range(n_pulse):
            pulse = pulses_smooth[i]
            baseline = baselines[i]
            max_val = max_vals[i]
            min_val = min_vals[i]
            if abs(max_val - baseline) > abs(min_val - baseline):
                # Positive pulse
                peak = max_val
                low_level = baseline + low_frac * (peak - baseline)
                high_level = baseline + high_frac * (peak - baseline)
                crossing_indices = np.where(pulse > low_level)[0]
                if len(crossing_indices) == 0:
                    continue  # leave as nan
                idx_low = crossing_indices[0]
                crossing_indices = np.where(pulse > high_level)[0]
                crossing_indices = crossing_indices[crossing_indices > idx_low]
                if len(crossing_indices) == 0:
                    continue
                idx_high = crossing_indices[0]
            else:
                # Negative pulse
                peak = min_val
                low_level = baseline + low_frac * (peak - baseline)
                high_level = baseline + high_frac * (peak - baseline)
                crossing_indices = np.where(pulse < low_level)[0]
                if len(crossing_indices) == 0:
                    continue
                idx_low = crossing_indices[0]
                crossing_indices = np.where(pulse < high_level)[0]
                crossing_indices = crossing_indices[crossing_indices > idx_low]
                if len(crossing_indices) == 0:
                    continue
                idx_high = crossing_indices[0]

            t_low[i] = t[idx_low]
            t_high[i] = t[idx_high]
            t_rise[i] = t_high[i] - t_low[i]

        return t_rise, t_low, t_high
 
    def find_decay_time(self, low_frac: float = 0.1, high_frac: float = 0.9, use_savgol: bool = True,
                    window_length: int = 51, polyorder: int = 3):
        """
        Find decay time between high_frac and low_frac thresholds for a batch of pulses.
        Returns arrays of (t_decay, t_high, t_low), each shape (n_pulse,)
        """
        data = self.data  # shape (n_pulse, n_samples)
        n_pulse, n_samples = data.shape
        t = self.time     # shape (n_samples,)

        # Optionally smooth data, shape (n_pulse, n_samples)
        if use_savgol:
            data_smooth = np.array([
                savgol_filter(pulse, window_length=window_length, polyorder=polyorder)
                for pulse in data
            ])
        else:
            data_smooth = data

        t_decay = np.full(n_pulse, np.nan)
        t_high = np.full(n_pulse, np.nan)
        t_low = np.full(n_pulse, np.nan)

        for i in range(n_pulse):
            pulse = data_smooth[i]
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

            idx_high_candidates = np.where(crossed(pulse[peak_idx:], high))[0]
            if idx_high_candidates.size == 0:
                continue
            idx_high = idx_high_candidates[0] + peak_idx

            idx_low_candidates = np.where(crossed(pulse[idx_high:], low))[0]
            if idx_low_candidates.size == 0:
                t_high[i] = t[idx_high]
                continue
            idx_low = idx_low_candidates[0] + idx_high

            t_decay[i] = t[idx_low] - t[idx_high]
            t_high[i] = t[idx_high]
            t_low[i] = t[idx_low]

        return t_decay, t_high, t_low

    def find_peak_fwhm_time(self, 
            L: float = 0.9, R: float = 0.9, type: str = 'peak', peak_deriv: bool = True, return_norm: bool = False, time_space: str = 'deriv',
            use_savgol: bool = True, window_length=51, polyorder=3, height=0.4, prominence=0.43):
        """
        Calculate the peak/FWHM time for a batch of pulses.
        Returns arrays of (interval, t_L, t_R) each shape (n_pulse,)
        """
        data = self.data  # shape: (n_pulse, n_samples)
        t = self.time     # shape: (n_samples,)
        n_pulse, n_samples = data.shape

        if type not in ('peak', 'fwhm'):
            raise TypeError("type must be 'peak' or 'fwhm'.")
        if type == 'fwhm':
            L = 0.5
            R = 0.5

        intervals = np.full(n_pulse, np.nan)
        t_Ls = np.full(n_pulse, np.nan)
        t_Rs = np.full(n_pulse, np.nan)

        for i in range(n_pulse):
            pulse = data[i]
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

            peaks, _ = find_peaks(np.abs(deriv_norm), height=height, prominence=prominence)
            if len(peaks) == 0:
                continue

            largest_peak_idx = peaks[np.argmax(np.abs(deriv_norm[peaks]))]
            sign = np.sign(deriv_norm[largest_peak_idx]) if deriv_norm[largest_peak_idx] != 0 else 1

            L_val = L * sign
            first_peak_idx = peaks[0]
            left = deriv_norm[:first_peak_idx+1]
            left_cross = np.where(sign * left >= sign * L_val)[0]
            idx_L = left_cross[0] if len(left_cross) > 0 else 0

            R_val = R * sign
            last_peak_idx = peaks[-1]
            right = deriv_norm[last_peak_idx:]
            right_cross = np.where(sign * right <= sign * R_val)[0]
            idx_R = right_cross[0] if len(right_cross) > 0 else len(t) - 1 - last_peak_idx

            # Interpolation
            def interp_time(idx, arr, val, t_arr, offset=0):
                if idx == 0:
                    return t_arr[offset + idx]
                x0, x1 = t_arr[offset + idx - 1], t_arr[offset + idx]
                y0, y1 = arr[idx - 1], arr[idx]
                if y1 == y0:
                    return x0
                return x0 + (val - y0) * (x1 - x0) / (y1 - y0)

            t_L_deriv = interp_time(idx_L, left, L_val, t)
            t_R_deriv = interp_time(idx_R, right, R_val, t, offset=last_peak_idx)

            if time_space == 'pulse':
                pulse_norm = (pulse - np.min(pulse)) / (np.max(pulse) - np.min(pulse))
                left_pulse = pulse_norm[:first_peak_idx+1]
                left_cross_pulse = np.where(sign * left_pulse >= sign * L)[0]
                idx_L_pulse = left_cross_pulse[0] if len(left_cross_pulse) > 0 else 0
                t_L = interp_time(idx_L_pulse, left_pulse, L, t)
                right_pulse = pulse_norm[last_peak_idx:]
                right_cross_pulse = np.where(sign * right_pulse <= sign * R)[0]
                idx_R_pulse = right_cross_pulse[0] if len(right_cross_pulse) > 0 else len(t) - 1 - last_peak_idx
                t_R = interp_time(idx_R_pulse, right_pulse, R, t, offset=last_peak_idx)
            else:
                t_L = t_L_deriv
                t_R = t_R_deriv

            if t_L < t_R:
                intervals[i] = abs(t_R - t_L)
                t_Ls[i] = t_L
                t_Rs[i] = t_R
            else:
                # Mark as invalid
                intervals[i] = np.nan
                t_Ls[i] = np.nan
                t_Rs[i] = np.nan

        if return_norm:
            # Optionally, also return the last computed deriv_norm (not batch, just for the last one)
            return intervals, t_Ls, t_Rs, deriv_norm
        else:
            return intervals, t_Ls, t_Rs

    def calculate_area(self):
        return np.sum(self.data, axis=1)

    def trapezoidal_filter(self, data, rise: float = 120, flat: float = 400):
        """
        Apply trapezoidal filter to a batch of pulses.
        :param data: 2D array (n_pulse, n_samples) or 1D array (n_samples)
        :param rise: rising length in samples
        :param flat: flat-top length in samples
        :return: filtered pulses, same shape as input
        """
        kernel = np.concatenate([np.ones(rise), np.zeros(flat), -np.ones(rise)])
        kernel = kernel / rise

        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            # For 2D, apply along axis 1 (samples)
            return np.array([np.convolve(row, kernel, mode='same') for row in data])

    def normalize_pulse(self, bl: float = 100, trap_rise: float = 120, trap_flat: float = 400, return_amp: bool = False):
        """
        Normalize a batch of pulses.
        :param bl: baseline length (default: 100)
        :param trap_rise: rising length in samples (default: 120)
        :param trap_flat: flat-top length in samples (default: 400)
        :return: normalized pulses, and optionally amplitudes
        """
        pulse = self.data  # shape (n_pulse, n_samples)
        if pulse.ndim == 1:
            pulse = pulse[np.newaxis, :]  # handle single pulse as batch

        # Baseline subtraction (per pulse)
        baselines = np.mean(pulse[:, :int(bl)], axis=1, keepdims=True)  # shape (n_pulse, 1)
        pulse_bs = pulse - baselines  # shape (n_pulse, n_samples)

        # Trapezoidal filter amplitude calculation (per pulse)
        filt_pulse = self.trapezoidal_filter(pulse_bs, rise=trap_rise, flat=trap_flat)
        start = int(trap_rise + trap_flat)
        amplitudes = np.max(np.abs(filt_pulse[:, start:]), axis=1)  # shape (n_pulse,)

        # Avoid division by zero
        amplitudes[amplitudes == 0] = 1

        # Normalize
        pulse_norm = pulse_bs / amplitudes[:, np.newaxis]  # shape (n_pulse, n_samples)
        if return_amp:
            return pulse_norm, amplitudes
        else:
            return pulse_norm

    def count_peaks(self, use_savgol: bool = True, deriv: bool = True, 
                window_length=51, polyorder=3, height=0.4, prominence=0.43):
        """
        Count peaks for each pulse in a batch.
        Returns an array of peak counts, one per pulse.
        """
        data = self.data  # shape (n_pulse, n_samples)
        n_pulse = data.shape[0]
        counts = np.zeros(n_pulse, dtype=int)

        for i in range(n_pulse):
            pulse = data[i]
            if deriv:
                if use_savgol:
                    pulse_proc = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
                else:
                    pulse_proc = np.gradient(pulse)
            else:
                if use_savgol:
                    pulse_proc = savgol_filter(pulse, window_length=window_length, polyorder=polyorder)
                else:
                    pulse_proc = pulse

            max_abs = np.max(np.abs(pulse_proc))
            if max_abs == 0:
                deriv_norm = pulse_proc * 0
            else:
                deriv_norm = pulse_proc / max_abs

            peaks, _ = find_peaks(np.abs(deriv_norm), height=height, prominence=prominence)
            counts[i] = len(peaks)
        return counts
    
    def l1_norm(self, reference_pulse):
        """
        Compute L1 norm between each pulse in the batch and the reference pulse.
        :param reference_pulse: PulseBatch or Pulse with .data and .time
        :return: array of L1 norms, shape (n_pulse,)
        """
        # Normalize both pulses
        pulse_norm = self.normalize_pulse()  # shape (n_pulse, n_samples)
        ref_norm = reference_pulse.normalize_pulse()
        t = self.time

        if pulse_norm.shape[1] != ref_norm.shape[-1]:
            raise ValueError("Pulses must have the same length.")

        if ref_norm.ndim == 1:
            ref_norm = np.broadcast_to(ref_norm, pulse_norm.shape)
        elif ref_norm.shape[0] != pulse_norm.shape[0]:
            # If reference is a batch but not same number of pulses, broadcast first pulse
            ref_norm = np.broadcast_to(ref_norm[0], pulse_norm.shape)

        # Find rising edge times on normalized pulses
        _, t_low_pulse, _ = self.find_rise_time()
        _, t_low_ref, _ = reference_pulse.find_rise_time()

        # Calculate shift (in samples) for each pulse
        dt = t[1] - t[0]
        shift_samples = (t_low_pulse - t_low_ref) / dt

        l1_norms = np.zeros(pulse_norm.shape[0])
        for i in range(pulse_norm.shape[0]):
            s = shift_samples[i]
            # Handle nan or inf shift
            if not np.isfinite(s):
                s = 0
            s = int(round(s))
            if s > 0:
                ref_aligned = np.pad(ref_norm[i], (s, 0), mode='constant')[:len(pulse_norm[i])]
                pulse_aligned = pulse_norm[i]
            elif s < 0:
                pulse_aligned = np.pad(pulse_norm[i], (-s, 0), mode='constant')[:len(ref_norm[i])]
                ref_aligned = ref_norm[i]
            else:
                pulse_aligned = pulse_norm[i]
                ref_aligned = ref_norm[i]

            min_len = min(len(pulse_aligned), len(ref_aligned))
            pulse_aligned = pulse_aligned[:min_len]
            ref_aligned = ref_aligned[:min_len]
            l1_norms[i] = np.sum(np.abs(pulse_aligned - ref_aligned))
        return l1_norms

    def find_amplitude(self, n_baseline: float = 100, rise: float = 120, flat: float = 400):
        """
        Find amplitude for each pulse in the batch using trapezoidal filter.
        Returns: amplitudes, shape (n_pulse,)
        """
        data = self.data  # shape (n_pulse, n_samples)
        n_pulse = data.shape[0]
        amplitudes = np.zeros(n_pulse)
        for i in range(n_pulse):
            baseline = np.mean(data[i, :int(n_baseline)])
            pulse_bs = data[i] - baseline
            filt = self.trapezoidal_filter(pulse_bs, rise, flat)
            start = int(rise + flat)
            amplitude = np.max(np.abs(filt[start:]))
            if amplitude == 0:
                amplitude = 1
            amplitudes[i] = amplitude
        return amplitudes

    def normalize_deriv(self, window_length: int = 51, polyorder: int = 3):
        """
        Return a normalized derivative for each pulse in a batch.
        Output: array of shape (n_pulse, n_samples)
        """
        data = self.data  # shape: (n_pulse, n_samples)
        derivs = []
        for pulse in data:
            deriv = savgol_filter(pulse, window_length=window_length, polyorder=polyorder, deriv=1)
            peak_idx = np.argmax(np.abs(deriv))
            peak_val = deriv[peak_idx]
            if peak_val == 0:
                peak_val = 1
            deriv_norm = deriv / peak_val
            derivs.append(deriv_norm)
        return np.stack(derivs)


class PulseGenerator:
    """
    PulseGenerator provides static and instance methods to generate synthetic pulses
    of various shapes (single-step, double-step, flat-top, etc.) and noise types,
    as well as to simulate batches of noisy and clean pulses for signal processing experiments.
    """

    def __init__(self, sampling_rate: float = 400e6):
        """
        Initialize the PulseGenerator.

        :param sampling_rate: Sampling rate in Hz (default: 400 MHz)
        """
        self.sampling_rate = sampling_rate

    @staticmethod
    def generate_single_step_pulse(time_vector, baseline, amplitude, t_start, steepness, decay_tau):
        """
        Generate a standard single-stage pulse (sigmoid rise and exponential decay).

        :param time_vector: Array of time points
        :param baseline: Baseline offset
        :param amplitude: Pulse amplitude
        :param t_start: Start time of the pulse
        :param steepness: Steepness of the rising edge
        :param decay_tau: Time constant for the exponential decay
        :return: 1D numpy array, the pulse
        """
        rising_edge = 1 / (1 + np.exp(-steepness * (time_vector - t_start)))
        decay_tail = np.exp(-np.maximum(0, time_vector - t_start) / decay_tau)
        return baseline + amplitude * rising_edge * decay_tail

    @staticmethod
    def generate_double_step_pulse(time_vector, baseline, amp1, t_start1, steepness1, amp2, t_start2, steepness2, decay_tau):
        """
        Generate a pulse with a two-stage rise (sum of two sigmoids) and exponential decay.

        :return: 1D numpy array, the pulse
        """
        step1 = amp1 / (1 + np.exp(-steepness1 * (time_vector - t_start1)))
        step2 = amp2 / (1 + np.exp(-steepness2 * (time_vector - t_start2)))
        combined_rise = step1 + step2
        decay_tail = np.exp(-np.maximum(0, time_vector - t_start1) / decay_tau)
        return baseline + combined_rise * decay_tail

    @staticmethod
    def generate_flat_top_pulse(time_vector, baseline, amplitude, t_start, rise_steepness, flat_duration, fall_steepness=None):
        """
        Generate a flat-top pulse: fast rise, flat region, and optional fall.

        :return: 1D numpy array, the pulse
        """
        rise = 1 / (1 + np.exp(-rise_steepness * (time_vector - t_start)))
        flat = (time_vector > t_start) & (time_vector < t_start + flat_duration)
        if fall_steepness is not None:
            t_fall = t_start + flat_duration
            fall = 1 / (1 + np.exp(fall_steepness * (time_vector - t_fall)))
            pulse = amplitude * rise * fall
        else:
            pulse = amplitude * rise
            pulse[time_vector > (t_start + flat_duration)] = amplitude
        return baseline + pulse

    @staticmethod
    def generate_wavy_noise(
        n_samples,
        white_noise_level,
        colored_noise_level,
        generate_ringing_noise=True,
        ringing_prob=0.5,
        max_ringing_bursts=3,
        ringing_amp_range=(0.05, 0.25),
        ringing_freq_range=(2, 12),
        ringing_damp_range=(0.008, 0.025),
        time_vector=None
    ):
        """
        Generate a noise trace containing white noise, colored noise, and optional ringing.

        :return: 1D numpy array, the noise trace
        """
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
                ringing[:t0] = 0
                ringing_noise += ringing

        return white_noise + colored_noise + ringing_noise

    def simulate_pulses(
        self,
        N_PULSES: int = 5000,
        N_SAMPLES: int = 4000,
        T_MAX: float = 2.6,
        PROB_NORMAL: float = 0.45,
        PROB_DOUBLE_STEP: float = 0.20,
        PROB_SLOW_RISER: float = 0.20,
        PROB_FLAT_TOP: float = 0.15,
        WHITE_NOISE_LEVEL: float = 0.03,
        COLORED_NOISE_LEVEL: float = 0.025,
        RINGING_NOISE: bool = False,
        RINGING_PROB: float = 0.7,
        RINGING_AMP_RANGE: tuple = (0.05, 0.25),
        RINGING_FREQ_RANGE: tuple = (2, 12),
        RINGING_DAMP_RANGE: tuple = (0.008, 0.025)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a batch of clean and noisy pulses of various types.

        :param N_PULSES: Number of pulses to generate.
        :param N_SAMPLES: Number of samples in each pulse.
        :param T_MAX: Maximum time value for the time vector (in microseconds).
        :param PROB_NORMAL: Probability of generating a standard single-step pulse.
        :param PROB_DOUBLE_STEP: Probability of generating a double-step pulse.
        :param PROB_SLOW_RISER: Probability of generating a slow-riser pulse.
        :param PROB_FLAT_TOP: Probability of generating a flat-top pulse.
        :param WHITE_NOISE_LEVEL: Standard deviation of the white noise added to each pulse.
        :param COLORED_NOISE_LEVEL: Scaling factor for the colored (integrated) noise.
        :param RINGING_NOISE: If True, include ringing noise in the generated pulses.
        :param RINGING_PROB: Probability that a given pulse will contain ringing noise.
        :param RINGING_AMP_RANGE: Range (min, max) of amplitudes for ringing noise bursts.
        :param RINGING_FREQ_RANGE: Range (min, max) of frequencies for ringing noise bursts.
        :param RINGING_DAMP_RANGE: Range (min, max) of damping factors for ringing noise bursts.

        :return: Tuple containing clean_pulses and noisy_pulses, each of shape (N_PULSES, N_SAMPLES).
        """
        time_vector = np.linspace(0, T_MAX, N_SAMPLES)
        clean_pulses = np.zeros((N_PULSES, N_SAMPLES))
        noisy_pulses = np.zeros((N_PULSES, N_SAMPLES))

        for i in range(N_PULSES):
            baseline = np.random.uniform(0, 0.25)
            decay_tau = np.random.uniform(20, 30)
            pulse_type = np.random.choice(
                ['normal', 'double_step', 'slow_riser', 'flat_top'],
                p=[PROB_NORMAL, PROB_DOUBLE_STEP, PROB_SLOW_RISER, PROB_FLAT_TOP]
            )

            if pulse_type == 'slow_riser':
                t_start1 = np.random.uniform(0.85, 1.05)
                steepness1 = np.random.uniform(5, 10)
                t_start2 = t_start1 + np.random.uniform(0.12, 0.17)
                while True:
                    amp1 = np.random.uniform(0.005, 0.2)
                    amp2 = np.random.uniform(0.01, 1.0 - amp1)
                    if amp1 + amp2 >= 0.015:
                        break
                steepness2 = np.random.uniform(30, 50)
                clean_pulse = PulseGenerator.generate_double_step_pulse(
                    time_vector, baseline,
                    amp1, t_start1, steepness1,
                    amp2, t_start2, steepness2,
                    decay_tau
                )
            elif pulse_type == 'double_step':
                t_start1 = np.random.uniform(0.8, 1.0)
                while True:
                    amp1 = np.random.uniform(0.005, 0.5)
                    amp2 = np.random.uniform(0.01, 1.0 - amp1)
                    if amp1 + amp2 >= 0.015:
                        break
                steepness1 = np.random.uniform(20, 50)
                t_start2 = t_start1 + np.random.uniform(0.1, 0.3)
                steepness2 = np.random.uniform(60, 120)
                clean_pulse = PulseGenerator.generate_double_step_pulse(
                    time_vector, baseline, amp1, t_start1, steepness1,
                    amp2, t_start2, steepness2, decay_tau
                )
            elif pulse_type == 'flat_top':
                amplitude = np.random.uniform(0.8, 1.0)
                t_start = np.random.uniform(1.0, 1.1)
                rise_steepness = np.random.uniform(8, 18)
                flat_duration = np.random.uniform(0.3, 0.7)
                clean_pulse = PulseGenerator.generate_flat_top_pulse(
                    time_vector, baseline, amplitude, t_start, rise_steepness, flat_duration
                )
            else:  # normal
                amplitude = np.random.uniform(0.015, 1.0)
                t_start = np.random.uniform(1.0, 1.1)
                steepness = np.random.uniform(8, 18)
                clean_pulse = PulseGenerator.generate_single_step_pulse(
                    time_vector, baseline, amplitude, t_start, steepness, decay_tau
                )

            noise = PulseGenerator.generate_wavy_noise(
                N_SAMPLES, WHITE_NOISE_LEVEL, COLORED_NOISE_LEVEL, generate_ringing_noise=RINGING_NOISE,
                ringing_prob=RINGING_PROB,
                ringing_amp_range=RINGING_AMP_RANGE,
                ringing_freq_range=RINGING_FREQ_RANGE,
                ringing_damp_range=RINGING_DAMP_RANGE,
                time_vector=time_vector
            )

            clean_pulses[i, :] = clean_pulse
            noisy_pulses[i, :] = clean_pulse + noise

        return clean_pulses, noisy_pulses


class DataParser:
    def __init__(self, header_file):
        """
        DataParser is used to parse CAEN binary data files and extract event information for a specified channel.

        Parameters
        ----------
        header_file : str
            Path to the binary header file. This file contains a 2-byte header word
            that encodes information about the data structure and endianness of all
            associated data files. The header is used to determine which data fields
            are present in the event records and how to unpack them.

        Example
        -------
        parser = DataParser(header_file="file.BIN")
        df = parser.save_channel_events(folder="data", channel=0, save_df=False)
        """
        self.header_file = header_file
        self.header_bits = self.get_caen_header_bits(header_file)
    
    def get_caen_header_bits(self, header_file):
        """
        Reads the header file and extracts bit flags and endianness
        for parsing binary event files.
        """
        with open(header_file, "rb") as f:
            raw_hdr = f.read(2)
            if len(raw_hdr) != 2:
                raise IOError("Header file is too short.")
            hdr_le = struct.unpack("<H", raw_hdr)[0]
            hdr_be = struct.unpack(">H", raw_hdr)[0]
            if (hdr_le & 0xFFF0) == 0xCAE0:
                header = hdr_le
                endian = "<"
            elif (hdr_be & 0xFFF0) == 0xCAE0:
                header = hdr_be
                endian = ">"
            else:
                raise ValueError("Header word does not match 0xCAE* pattern.")
            energy_ch      = bool(header & 0x1)
            energy_cal     = bool(header & 0x2)
            energy_short   = bool(header & 0x4)
            waveform_field = bool(header & 0x8)
            return dict(
                energy_ch=energy_ch,
                energy_cal=energy_cal,
                energy_short=energy_short,
                waveform_field=waveform_field,
                endian=endian
            )
    
    def parse_events_channel(self, data_file, channel=0):
        """
        Parses a single binary data file and extracts event information for the specified channel.
        """
        events = []
        energy_ch      = self.header_bits['energy_ch']
        energy_cal     = self.header_bits['energy_cal']
        energy_short   = self.header_bits['energy_short']
        waveform_field = self.header_bits['waveform_field']
        endian         = self.header_bits['endian']

        with open(data_file, "rb") as f:
            # Check for header in this file
            raw_hdr = f.read(2)
            possible_header = struct.unpack(endian + "H", raw_hdr)[0]
            if (possible_header & 0xFFF0) == 0xCAE0:
                # Header present, already read
                pass
            else:
                f.seek(0)

            while True:
                board_bytes = f.read(2)
                if not board_bytes or len(board_bytes) < 2:
                    break
                board = struct.unpack(endian + "H", board_bytes)[0]
                channel_bytes = f.read(2)
                if not channel_bytes or len(channel_bytes) < 2:
                    break
                evt_channel = struct.unpack(endian + "H", channel_bytes)[0]
                timestamp_bytes = f.read(8)
                if not timestamp_bytes or len(timestamp_bytes) < 8:
                    break
                timestamp = struct.unpack(endian + "Q", timestamp_bytes)[0]

                # Defaults
                e_ch = None
                e_cal = None
                e_short = None
                pulse = None
                wcode = None

                if energy_ch:
                    e_ch_bytes = f.read(2)
                    if len(e_ch_bytes) < 2:
                        break
                    e_ch = struct.unpack(endian + "H", e_ch_bytes)[0]
                if energy_cal:
                    e_cal_bytes = f.read(8)
                    if len(e_cal_bytes) < 8:
                        break
                    e_cal = struct.unpack(endian + "d", e_cal_bytes)[0]
                if energy_short:
                    e_short_bytes = f.read(2)
                    if len(e_short_bytes) < 2:
                        break
                    e_short = struct.unpack(endian + "H", e_short_bytes)[0]

                if len(f.read(4)) < 4: break  # flags

                if waveform_field:
                    wcode_bytes = f.read(1)
                    if not wcode_bytes or len(wcode_bytes) < 1:
                        break
                    wcode = struct.unpack(endian + "B", wcode_bytes)[0]
                    nsamples_bytes = f.read(4)
                    if not nsamples_bytes or len(nsamples_bytes) < 4:
                        break
                    nsamples = struct.unpack(endian + "I", nsamples_bytes)[0]
                    samples_bytes = f.read(2 * nsamples)
                    if not samples_bytes or len(samples_bytes) < 2 * nsamples:
                        break
                    pulse = list(struct.unpack(endian + f"{nsamples}H", samples_bytes))

                if evt_channel == channel and wcode == 1:
                    events.append({
                        'channel': evt_channel,
                        'timestamp': timestamp,
                        'pulse': pulse,            # bit 3
                        'energy_ch': e_ch,         # bit 0
                        'energy_cal': e_cal,       # bit 1
                        'energy_short': e_short    # bit 2
                    })

        return events

    def save_events(self, folder: str, channel: int, n_chunks: int = 5, out_prefix: str = "data", save_df: bool = True):
        """
        Parse all BIN files in the folder, extract events for the given channel,
        and return a single DataFrame. Optionally save the DataFrame to disk.

        Parameters
        ----------
        folder : str
            Path to the folder containing .BIN files.
        channel : int
            Channel number to extract events for.
        n_chunks : int, optional
            Number of chunks to split the files for progress reporting (default=5).
        out_prefix : str, optional
            Prefix for the output file name (default="data_2025").
        save_df : bool, optional
            If True, save the DataFrame as a compressed pickle file (default=True).

        Returns
        -------
        pd.DataFrame
            DataFrame containing all events for the specified channel.

        Notes
        -----
        The header_file provided when constructing the DataParser instance must be a
        valid CAEN header file (2 bytes) that matches the data format of the .BIN files.
        """
        files = sorted(
            glob.glob(os.path.join(folder, "*.BIN")) + 
            glob.glob(os.path.join(folder, "*.bin"))
        )
        print('number of .BIN files:', len(files))
        if not files:
            print("No .BIN files found.")
            return pd.DataFrame(columns=['channel', 'timestamp', 'pulse', 'energy_ch', 'energy_cal', 'energy_short'])

        chunk_size = math.ceil(len(files) / n_chunks)
        all_events = []
        for chunk_idx in range(n_chunks):
            chunk_files = files[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]
            for fname in chunk_files:
                events = self.parse_events_channel(fname, channel=channel)
                all_events.extend(events)
        df = pd.DataFrame(
            all_events, 
            columns=['channel', 'timestamp', 'pulse', 'energy_ch', 'energy_cal', 'energy_short']
        )
        if save_df:
            out_dir = os.path.join(folder, f'channel{channel}')
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"{out_prefix}_ch{channel}.pkl.gz")
            df.to_pickle(filename, compression="gzip")
            print(f"Saved DataFrame with {len(files)} files and {len(df)} events.")
        return df