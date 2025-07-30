import numpy as np

import os, sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from pulse import Pulse
from pulse_batch import PulseBatch

def test_pulsebatch():
    data = np.array([[1, 2, 3, 4, 5],
                     [5, 4, 3, 2, 1]])
    pulses = PulseBatch(data)

    assert np.array_equal(pulses.data, data)
    assert pulses.sampling_rate == 400e6
    assert pulses.label is None
    assert pulses.data.shape == (2, 5)
    assert len(pulses.time) == len(data[0])
    assert pulses.time[0] == 0

    # Test with custom parameters
    pulses2 = PulseBatch(data, sampling_rate=500e6, label="test")
    assert pulses2.sampling_rate == 500e6
    assert pulses2.label == "test"

def test_find_rise_time():
    data = np.stack([
        np.concatenate([
            np.zeros(200),
            np.linspace(0, 1, 100),
            np.ones(624)
        ]),
        np.concatenate([
            np.zeros(100),
            np.linspace(0, 2, 200),
            np.ones(624)*2
        ])
    ])
    pulses = PulseBatch(data)

    rise_time, t_low, t_high = pulses.find_rise_times()
    assert rise_time.shape == (2,)
    assert t_low.shape == (2,)
    assert t_high.shape == (2,)
    assert np.all(rise_time > 0)
    assert np.all(t_low < t_high)

    rise_time2, t_low2, t_high2 = pulses.find_rise_times(use_savgol=False)
    assert rise_time2.shape == (2,)
    assert np.all(rise_time2 > 0)
    assert np.all(t_low2 < t_high2)

def test_find_decay_time():
    # Create two simple falling (decay) pulses
    pulse1 = np.concatenate([
        np.zeros(200),           # baseline
        np.ones(100),            # plateau
        np.linspace(1, 0, 100),  # falling edge (decay)
        np.zeros(624)            # return to baseline
    ])
    pulse2 = np.concatenate([
    np.zeros(200),           # baseline
    np.ones(100),            # plateau
    np.linspace(2, 0, 100),  # falling edge (same length as pulse1)
    np.zeros(624)            # return to baseline
    ])
    data = np.stack([pulse1, pulse2])
    pulses = PulseBatch(data)  # assumes PulseBatch auto-generates self.time

    # Test with smoothing
    decay_time, t_high, t_low = pulses.find_decay_times()
    assert decay_time.shape == (2,)
    assert t_high.shape == (2,)
    assert t_low.shape == (2,)
    assert np.all(decay_time > 0)
    assert np.all(t_high < t_low)

    # Test without smoothing
    decay_time2, t_high2, t_low2 = pulses.find_decay_times(use_savgol=False)
    assert decay_time2.shape == (2,)
    assert np.all(decay_time2 > 0)
    assert np.all(t_high2 < t_low2)

def test_find_peak_fwhm_time():
    n_points = 1024
    t = np.linspace(0, 1, n_points)
    data1 = 1 / (1 + np.exp(-40 * (t - 0.4))) - 1 / (1 + np.exp(-40 * (t - 0.6)))
    data2 = 1 / (1 + np.exp(-20 * (t - 0.3))) - 1 / (1 + np.exp(-20 * (t - 0.7)))
    data = np.stack([data1, data2])
    pulses = PulseBatch(data)

    # Default parameters
    interval, t_L, t_R = pulses.find_peak_fwhm_times()
    assert interval.shape == (2,)
    valid = (~np.isnan(interval)) & (interval > 0) & (t_L < t_R)
    assert np.any(valid), "At least one valid interval should be found (default params)"
    assert np.all(interval[valid] > 0)
    assert np.all(t_L[valid] < t_R[valid])

    # FWHM
    interval_fwhm, t_L_f, t_R_f = pulses.find_peak_fwhm_times(type='fwhm')
    valid_fwhm = (~np.isnan(interval_fwhm)) & (interval_fwhm > 0) & (t_L_f < t_R_f)
    assert np.any(valid_fwhm), "At least one valid interval should be found (FWHM)"
    assert np.all(interval_fwhm[valid_fwhm] > 0)
    assert np.all(t_L_f[valid_fwhm] < t_R_f[valid_fwhm])

    # Pulse space (handle possibly invalid intervals)
    interval_pulse, t_L_p, t_R_p = pulses.find_peak_fwhm_times(time_space='pulse')
    valid_pulse = (~np.isnan(interval_pulse)) & (interval_pulse > 0) & (t_L_p < t_R_p)
    assert np.any(valid_pulse), "At least one valid interval should be found (pulse space)"
    assert np.all((t_L_p[~np.isnan(interval_pulse)] < t_R_p[~np.isnan(interval_pulse)]))
    assert np.all(interval_pulse[valid_pulse] > 0)
    assert np.all(t_L_p[valid_pulse] < t_R_p[valid_pulse])

    # No smoothing
    interval_raw, t_L_raw, t_R_raw = pulses.find_peak_fwhm_times(use_savgol=False)
    valid_raw = (~np.isnan(interval_raw)) & (interval_raw > 0) & (t_L_raw < t_R_raw)
    assert np.any(valid_raw), "At least one valid interval should be found (no smoothing)"
    assert np.all(interval_raw[valid_raw] > 0)
    assert np.all(t_L_raw[valid_raw] < t_R_raw[valid_raw])

def test_calculate_area():
    # Constant pulses: area should be number of samples for each
    data = np.stack([
        np.ones(100),         # area = 100
        np.ones(100)*2        # area = 200
    ])
    pulses = PulseBatch(data)
    area = pulses.calculate_area()
    assert area.shape == (2,)
    assert np.allclose(area, [100, 200])

    # Two ramps
    data = np.stack([
        np.arange(10),        # area = 45
        np.arange(10, 20)     # area = sum 10..19 = 145
    ])
    pulses = PulseBatch(data)
    area = pulses.calculate_area()
    assert np.allclose(area, [45, 145])

    # Negative pulses
    data = np.stack([
        -np.ones(50),         # area = -50
        -np.ones(50)*2        # area = -100
    ])
    pulses = PulseBatch(data)
    area = pulses.calculate_area()
    assert np.allclose(area, [-50, -100])

    # Pulses with zeros
    data = np.stack([
        np.array([0, 0, 1, 1, 0, 0]),   # area = 2
        np.zeros(6)                     # area = 0
    ])
    pulses = PulseBatch(data)
    area = pulses.calculate_area()
    assert np.allclose(area, [2, 0])

def test_normalize_pulse():
    # Create a batch of pulses with various known shapes
    data1 = np.concatenate([np.zeros(100), np.ones(50)*5, np.zeros(100)])
    data2 = np.zeros(200)  # all zeros
    data3 = np.concatenate([np.zeros(100), -np.ones(50)*4, np.zeros(100)])
    data4 = np.ones(50)    # short pulse (shorter than bl)
    all_data = [data1, data2, data3, data4]
    maxlen = max(len(d) for d in all_data)
    data_padded = [np.pad(d, (0, maxlen - len(d))) for d in all_data]
    data = np.stack(data_padded)

    pulses = PulseBatch(data)
    norm_pulse = pulses.normalize_pulses()

    # Check each pulse individually
    for i, orig_data in enumerate(data):
        baseline = np.mean(orig_data[:100])
        pulse_bs = orig_data - baseline
        filt_pulse = Pulse(pulse_bs).trapezoidal_filter(pulse_bs,rise=120, flat=400)
        start = 120 + 400
        # Fix: handle short pulses
        if start < len(filt_pulse):
            amplitude = np.max(np.abs(filt_pulse[start:]))
        else:
            amplitude = np.max(np.abs(filt_pulse))
        if amplitude == 0:
            amplitude = 1
        expected_norm_pulse = pulse_bs / amplitude

        assert np.allclose(norm_pulse[i], expected_norm_pulse)
        assert np.isclose(np.max(np.abs(norm_pulse[i])), np.max(np.abs(expected_norm_pulse)), atol=1e-6)

    # Test: return_amp returns both normalized pulse and amplitude
    norm_pulse2, amp = pulses.normalize_pulses(return_amp=True)
    for i, orig_data in enumerate(data):
        baseline = np.mean(orig_data[:100])
        pulse_bs = orig_data - baseline
        filt_pulse = Pulse(pulse_bs).trapezoidal_filter(pulse_bs,rise=120, flat=400)
        start = 120 + 400
        if start < len(filt_pulse):
            amplitude = np.max(np.abs(filt_pulse[start:]))
        else:
            amplitude = np.max(np.abs(filt_pulse))
        if amplitude == 0:
            amplitude = 1
        expected_norm_pulse = pulse_bs / amplitude
        assert np.allclose(norm_pulse2[i], expected_norm_pulse)
        assert np.isclose(amp[i], amplitude)

def test_count_peaks():
    # Case 1: Single step pulse (should have two derivative peaks: rise and fall)
    data1 = np.concatenate([np.zeros(100), np.ones(50), np.zeros(100)])  # 2 peaks

    # Case 2: Two blocks (should have four derivative peaks)
    data2 = np.concatenate([np.zeros(100), np.ones(30), np.zeros(50), np.ones(30), np.zeros(100)])  # 4 peaks

    # Case 3: No peaks (flat signal)
    data3 = np.zeros(200)  # 0 peaks

    # Case 4: Peaks without smoothing
    data4 = np.concatenate([np.zeros(50), np.ones(10), np.zeros(50), np.ones(10), np.zeros(50)])  # 4 peaks

    # Pad all to the same length
    all_data = [data1, data2, data3, data4]
    maxlen = max(len(d) for d in all_data)
    data_padded = [np.pad(d, (0, maxlen - len(d))) for d in all_data]
    data = np.stack(data_padded)

    pulses = PulseBatch(data)
    n_peaks = pulses.count_peaks()
    assert n_peaks[0] == 2
    assert n_peaks[1] == 4
    assert n_peaks[2] == 0

    # Test: Peaks without smoothing (use_savgol=False)
    n_peaks_no_smooth = pulses.count_peaks(use_savgol=False)
    assert n_peaks_no_smooth[3] == 4

def test_l1_norm():
    # Create a batch of identical pulses
    data = np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.ones(100)])
    data_batch = np.stack([data, data])
    pulses = PulseBatch(data_batch)

    # Reference: single pulse
    ref_pulse = Pulse(data_batch[0])

    # L1 norm of a pulse batch with a single identical pulse should be zero (or very close)
    l1 = pulses.l1_norm(ref_pulse)
    assert l1.shape == (2,)
    assert np.allclose(l1, 0, atol=1e-10)

    # L1 norm with a shifted pulse batch should be > 0
    shifted_data = np.concatenate([np.zeros(105), np.linspace(0, 1, 50), np.ones(95)])
    shifted_batch = np.stack([shifted_data, shifted_data])
    pulses_shifted = PulseBatch(shifted_batch)
    l1_shifted = pulses_shifted.l1_norm(ref_pulse)
    assert np.all(l1_shifted > 0)

    # L1 norm with a different shaped pulse batch should be > 0
    ref_pulse = Pulse(np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.ones(100)]))
    data_diff = np.concatenate([np.zeros(100), np.linspace(0, 2, 50), np.ones(100)])
    diff_batch = np.stack([data_diff, data_diff])
    pulses_diff = PulseBatch(diff_batch)
    l1_diff = pulses_diff.l1_norm(ref_pulse)
    print(l1_diff)
    assert np.all(l1_diff > 0)

    # L1 norm with a negative pulse batch should be > 0
    data_neg = -data
    neg_batch = np.stack([data_neg, data_neg])
    pulses_neg = PulseBatch(neg_batch)
    l1_neg = pulses_neg.l1_norm(ref_pulse)
    assert np.all(l1_neg > 0)

    # L1 norm with different lengths should raise ValueError
    data_short = np.zeros(50)
    short_batch = np.stack([data_short, data_short])
    pulses_short = PulseBatch(short_batch)
    try:
        _ = pulses_short.l1_norm(ref_pulse)
        assert False, "Expected ValueError for different lengths"
    except ValueError:
        pass

def test_find_amplitude():
    data1 = np.concatenate([np.zeros(100), np.ones(50)*5, np.zeros(100)])      # positive pulse
    data2 = np.concatenate([np.zeros(100), -np.ones(50)*3, np.zeros(100)])     # negative pulse
    data3 = np.zeros(200)                                                      # all zeros
    all_data = [data1, data2, data3]
    maxlen = max(len(d) for d in all_data)
    data_padded = [np.pad(d, (0, maxlen - len(d))) for d in all_data]
    data = np.stack(data_padded)

    pulses = PulseBatch(data)
    amplitudes = pulses.find_amplitudes()
    assert amplitudes.shape == (3,)
    assert amplitudes[0] > 0
    assert amplitudes[1] > 0
    assert amplitudes[2] == 1  # as per code logic

def test_normalize_deriv():
    # Pulse with a positive and negative ramp
    data1 = np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.linspace(1, 0, 50), np.zeros(100)])
    # All zeros
    data2 = np.zeros(200)
    # Negative ramp
    data3 = np.concatenate([np.zeros(100), np.linspace(0, -1, 50), np.linspace(-1, 0, 50), np.zeros(100)])

    # Pad all to the same length
    all_data = [data1, data2, data3]
    maxlen = max(len(d) for d in all_data)
    data_padded = [np.pad(d, (0, maxlen - len(d))) for d in all_data]
    data = np.stack(data_padded)

    pulses = PulseBatch(data)
    norm_derivs = pulses.normalize_deriv(window_length=11)

    # Main peak in derivative should be +1 or -1 for nonzero pulses
    max_abs = np.max(np.abs(norm_derivs), axis=1)
    for i, orig_data in enumerate(data_padded):
        if np.any(orig_data != 0):
            assert np.isclose(max_abs[i], 1, atol=1e-2)
        else:
            assert np.allclose(norm_derivs[i], 0)

    # Check only the central part of the flat region, far from edges
    assert np.allclose(norm_derivs[0, 20:80], 0, atol=0.02)
    assert np.allclose(norm_derivs[0, -80:-20], 0, atol=0.02)
    assert np.allclose(norm_derivs[1], 0)
    assert np.isclose(np.max(np.abs(norm_derivs[2])), 1, atol=1e-2)

if __name__ == "__main__":
    test_pulsebatch()
    test_find_rise_time()
    test_find_decay_time()
    test_find_peak_fwhm_time()
    test_calculate_area()
    test_normalize_pulse()
    test_count_peaks()
    test_l1_norm()
    test_find_amplitude()
    test_normalize_deriv()