import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple
import struct
import os
import glob
import math
import pandas as pd
from pulse import Pulse


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

    def find_rise_times(self, **kwargs):
        """
        Compute rise times for all pulses in the batch.
        Returns:
            rise_times: np.ndarray of shape (n_pulses,)
            t_lows: np.ndarray of shape (n_pulses,)
            t_highs: np.ndarray of shape (n_pulses,)
        """
        rise_times = []
        t_lows = []
        t_highs = []

        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            t_rise, t_low, t_high = pulse.find_rise_time(**kwargs)
            rise_times.append(t_rise)
            t_lows.append(t_low)
            t_highs.append(t_high)

        return np.array(rise_times), np.array(t_lows), np.array(t_highs)
 
    def find_decay_times(self, **kwargs):
        """
        Compute decay times for all pulses in the batch.
        Returns:
            decay_times: np.ndarray of shape (n_pulses,)
            t_highs: np.ndarray of shape (n_pulses,)
            t_lows: np.ndarray of shape (n_pulses,)
        """
        decay_times = []
        t_highs = []
        t_lows = []

        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            t_decay, t_high, t_low = pulse.find_decay_time(**kwargs)
            decay_times.append(t_decay)
            t_highs.append(t_high)
            t_lows.append(t_low)

        return np.array(decay_times), np.array(t_highs), np.array(t_lows)

    def find_peak_fwhm_times(self, **kwargs):
        """
        Calculate peak/FWHM times for all pulses in the batch.
        Returns:
            intervals: np.ndarray of shape (n_pulses,)
            t_Ls: np.ndarray of shape (n_pulses,)
            t_Rs: np.ndarray of shape (n_pulses,)
            (optionally) normed: list of np.ndarrays (if return_norm=True)
        """
        intervals = []
        t_Ls = []
        t_Rs = []
        normed = []

        # Check if user wants normalized output
        want_norm = kwargs.get('return_norm', False)

        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            result = pulse.find_peak_fwhm_time(**kwargs)
            if want_norm:
                interval, t_L, t_R, norm = result
                normed.append(norm)
            else:
                interval, t_L, t_R = result
            intervals.append(interval)
            t_Ls.append(t_L)
            t_Rs.append(t_R)

        if want_norm:
            return np.array(intervals), np.array(t_Ls), np.array(t_Rs), normed
        else:
            return np.array(intervals), np.array(t_Ls), np.array(t_Rs)

    def calculate_area(self):
        return np.sum(self.data, axis=1)

    def trapezoidal_filter(self, rise=120, flat=400):
        """
        Apply trapezoidal filter to all pulses in the batch.
        Returns a numpy array of shape (n_pulses, n_samples)
        """
        filtered = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            filtered.append(pulse.trapezoidal_filter(pulse.data, rise=rise, flat=flat))
        return np.array(filtered)

    def normalize_pulses(self, bl=100, trap_rise=120, trap_flat=400, return_amp=False):
        """
        Normalize all pulses in the batch.
        Returns:
            - pulse_norms: (n_pulses, n_samples)
            - amplitudes: (n_pulses,) if return_amp is True
        """
        pulse_norms = []
        amplitudes = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            result = pulse.normalize_pulse(bl=bl, trap_rise=trap_rise, trap_flat=trap_flat, return_amp=return_amp)
            if return_amp:
                norm, amp = result
                pulse_norms.append(norm)
                amplitudes.append(amp)
            else:
                norm = result
                pulse_norms.append(norm)
        if return_amp:
            return np.array(pulse_norms), np.array(amplitudes)
        else:
            return np.array(pulse_norms)

    def count_peaks(self, **kwargs):
        """
        Count peaks for each pulse in the batch using the Pulse class's count_peaks method.
        Returns:
            counts: np.ndarray of shape (n_pulses,)
        """
        counts = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            n_peaks = pulse.count_peaks(**kwargs)
            counts.append(n_peaks)
        return np.array(counts)
    
    def l1_norm(self, reference_pulse):
        """
        Compute L1 norm (with alignment) between each pulse in the batch and a reference pulse.
        :param reference_pulse: A Pulse object to compare against
        :return: np.ndarray of L1 norms, shape (n_pulses,)
        """
        l1s = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            l1 = pulse.l1_norm(reference_pulse)
            l1s.append(l1)
        return np.array(l1s)

    def find_amplitudes(self, n_baseline: float = 100, rise: float = 120, flat: float = 400):
        """
        Find amplitudes for all pulses in the batch using the per-pulse find_amplitude method.
        Returns:
            amplitudes: np.ndarray of shape (n_pulses,)
        """
        amplitudes = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            amp = pulse.find_amplitude(n_baseline=n_baseline, rise=rise, flat=flat)
            amplitudes.append(amp)
        return np.array(amplitudes)

    def normalize_deriv(self, window_length: int = 51, polyorder: int = 3):
        """
        Apply normalize_deriv to all pulses in the batch.
        Returns:
            deriv_norms: np.ndarray of shape (n_pulses, n_samples)
        """
        deriv_norms = []
        for i in range(self.data.shape[0]):
            pulse = Pulse(self.data[i], self.sampling_rate)
            norm = pulse.normalize_deriv(window_length=window_length, polyorder=polyorder)
            deriv_norms.append(norm)
        return np.array(deriv_norms)


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
        df = parser.save_events(folder="data", channel=0, save_df=False)
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