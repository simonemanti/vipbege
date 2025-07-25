import numpy as np

from vipbege.pulse import Pulse

def test_pulse():
    # Test with default sampling rate
    data = np.array([1, 2, 3, 4, 5])
    pulse = Pulse(data)
    
    assert np.array_equal(pulse.data, data)
    assert pulse.sampling_rate == 400e6
    assert pulse.label is None
    assert len(pulse.time) == len(data)
    assert pulse.time[0] == 0
    
    # Test with custom parameters
    pulse2 = Pulse(data, sampling_rate=500e6, label="test")
    assert pulse2.sampling_rate == 500e6
    assert pulse2.label == "test"

def test_find_rise_time():
    # Create a simple step pulse
    data = np.concatenate([
        np.zeros(200),      # baseline
        np.linspace(0, 1, 100),  # rising edge
        np.ones(624)        # plateau
    ])
    
    pulse = Pulse(data)
    
    # Test with smoothing
    rise_time, t_low, t_high = pulse.find_rise_time()
    assert rise_time > 0
    assert t_low < t_high
    
    # Test without smoothing
    rise_time2, t_low2, t_high2 = pulse.find_rise_time(use_savgol=False)
    assert rise_time2 > 0
    assert t_low2 < t_high2

def test_find_decay_time():
    # Create a simple falling (decay) pulse
    data = np.concatenate([
        np.zeros(200),           # baseline
        np.ones(100),            # plateau
        np.linspace(1, 0, 100),  # falling edge (decay)
        np.zeros(624)            # return to baseline
    ])
    pulse = Pulse(data)  # Assumes Pulse auto-generates self.time if not given

    # Test with smoothing (if your method supports it)
    decay_time, t_high, t_low = pulse.find_decay_time()
    assert decay_time > 0
    assert t_high < t_low

    # Optionally: Test without smoothing, if your method supports a flag
    decay_time2, t_high2, t_low2 = pulse.find_decay_time(use_savgol=False)
    assert decay_time2 > 0
    assert t_high2 < t_low2

def test_find_peak_fwhm_time():
    # Create a synthetic sigmoid-like pulse
    n_points = 1024
    t = np.linspace(0, 1, n_points)
    data = 1 / (1 + np.exp(-40 * (t - 0.4))) - 1 / (1 + np.exp(-40 * (t - 0.6)))
    pulse = Pulse(data)  # No time argument needed

    # Test default parameters (should find peak time at the center)
    interval, t_L, t_R = pulse.find_peak_fwhm_time()
    assert interval > 0
    assert t_L < t_R

    # Test FWHM mode
    interval_fwhm, t_L_f, t_R_f = pulse.find_peak_fwhm_time(type='fwhm')
    assert interval_fwhm > 0
    assert t_L_f < t_R_f

    # Test in pulse space
    interval_pulse, t_L_p, t_R_p = pulse.find_peak_fwhm_time(time_space='pulse')
    assert interval_pulse > 0
    assert t_L_p < t_R_p

    # Test without Savitzky-Golay smoothing
    interval_raw, t_L_raw, t_R_raw = pulse.find_peak_fwhm_time(use_savgol=False)
    assert interval_raw > 0
    assert t_L_raw < t_R_raw

def test_calculate_area():
    # Simple known case: area under a constant pulse
    data = np.ones(100)  # Area should be 100
    pulse = Pulse(data)
    area = pulse.calculate_area()
    assert area == 100

    # Area under a ramp
    data = np.arange(10)  # Area should be sum 0+1+...+9 = 45
    pulse = Pulse(data)
    area = pulse.calculate_area()
    assert area == 45

    # Area under a negative pulse
    data = -np.ones(50)   # Area should be -50
    pulse = Pulse(data)
    area = pulse.calculate_area()
    assert area == -50

    # Area under a pulse with zeros
    data = np.array([0, 0, 1, 1, 0, 0])
    pulse = Pulse(data)
    area = pulse.calculate_area()
    assert area == 2

def test_normalize_pulse():
    # Create a simple pulse with a known shape
    data = np.concatenate([np.zeros(100), np.ones(50)*5, np.zeros(100)])
    pulse = Pulse(data)
    norm_pulse = pulse.normalize_pulse()
    
    # Compute expected amplitude using the filter
    baseline = np.mean(data[:100])
    pulse_bs = data - baseline
    filt_pulse = pulse.trapezoidal_filter(pulse_bs, rise=120, flat=400)
    start = 120 + 400
    amplitude = np.max(np.abs(filt_pulse[start:]))
    if amplitude == 0:
        amplitude = 1
    expected_norm_pulse = pulse_bs / amplitude

    # Check normalization against expected
    assert np.allclose(norm_pulse, expected_norm_pulse)
    assert np.isclose(np.max(np.abs(norm_pulse)), np.max(np.abs(expected_norm_pulse)), atol=1e-6)

    # Test: return_amp returns both normalized pulse and amplitude
    norm_pulse2, amp = pulse.normalize_pulse(return_amp=True)
    assert np.allclose(norm_pulse2, expected_norm_pulse)
    assert amp == amplitude

    # Test: All zeros input
    data_zeros = np.zeros(200)
    pulse_zeros = Pulse(data_zeros)
    norm_zeros = pulse_zeros.normalize_pulse()
    assert np.allclose(norm_zeros, 0)

    # Test: Negative pulse
    data_neg = np.concatenate([np.zeros(100), -np.ones(50)*4, np.zeros(100)])
    pulse_neg = Pulse(data_neg)
    norm_neg = pulse_neg.normalize_pulse()
    baseline_neg = np.mean(data_neg[:100])
    pulse_bs_neg = data_neg - baseline_neg
    filt_pulse_neg = pulse_neg.trapezoidal_filter(pulse_bs_neg, rise=120, flat=400)
    amplitude_neg = np.max(np.abs(filt_pulse_neg[start:]))
    if amplitude_neg == 0:
        amplitude_neg = 1
    expected_norm_neg = pulse_bs_neg / amplitude_neg
    assert np.allclose(norm_neg, expected_norm_neg)

    # Test: Short pulse (shorter than bl)
    data_short = np.ones(50)
    pulse_short = Pulse(data_short)
    norm_short = pulse_short.normalize_pulse(bl=60)
    # Baseline subtraction will make all zeros
    assert np.allclose(norm_short, 0)

def test_count_peaks():
    # Case 1: Single step pulse (should have two derivative peaks: rise and fall)
    data = np.concatenate([np.zeros(100), np.ones(50), np.zeros(100)])
    pulse = Pulse(data)
    n_peaks = pulse.count_peaks()
    assert n_peaks == 2

    # Case 2: Two blocks (should have four derivative peaks)
    data2 = np.concatenate([np.zeros(100), np.ones(30), np.zeros(50), np.ones(30), np.zeros(100)])
    pulse2 = Pulse(data2)
    n_peaks2 = pulse2.count_peaks()
    assert n_peaks2 == 4

    # Case 3: No peaks (flat signal)
    data3 = np.zeros(200)
    pulse3 = Pulse(data3)
    n_peaks3 = pulse3.count_peaks()
    assert n_peaks3 == 0

    # Case 4: Peaks without smoothing
    data5 = np.concatenate([np.zeros(50), np.ones(10), np.zeros(50), np.ones(10), np.zeros(50)])
    pulse5 = Pulse(data5)
    n_peaks5 = pulse5.count_peaks(use_savgol=False)
    assert n_peaks5 == 4

def test_l1_norm():
    # Create a synthetic pulse
    data = np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.ones(100)])
    pulse1 = Pulse(data)
    pulse2 = Pulse(data.copy())  # Identical pulse

    # L1 norm of a pulse with itself should be zero (or extremely close to zero)
    l1 = pulse1.l1_norm(pulse2)
    assert np.isclose(l1, 0, atol=1e-10)

    # L1 norm with a shifted pulse should be > 0
    shifted_data = np.concatenate([np.zeros(105), np.linspace(0, 1, 50), np.ones(95)])
    pulse_shifted = Pulse(shifted_data)
    l1_shifted = pulse1.l1_norm(pulse_shifted)
    assert l1_shifted > 0

    # L1 norm with a different shaped pulse should be > 0
    data_diff = np.concatenate([np.zeros(100), np.linspace(0, 2, 50), np.ones(100)])
    pulse_diff = Pulse(data_diff)
    l1_diff = pulse1.l1_norm(pulse_diff)
    assert l1_diff > 0

    # L1 norm with a negative pulse
    data_neg = -data
    pulse_neg = Pulse(data_neg)
    l1_neg = pulse1.l1_norm(pulse_neg)
    assert l1_neg > 0

    # L1 norm with different lengths should raise ValueError
    data_short = np.zeros(50)
    pulse_short = Pulse(data_short)
    try:
        _ = pulse1.l1_norm(pulse_short)
        assert False, "Expected ValueError for different lengths"
    except ValueError:
        pass

def test_find_amplitude():
    data = np.concatenate([np.zeros(100), np.ones(50)*5, np.zeros(100)])
    pulse = Pulse(data)
    amplitude = pulse.find_amplitude()
    # Just check amplitude is positive and nonzero (filter changes value)
    assert amplitude > 0

    data_neg = np.concatenate([np.zeros(100), -np.ones(50)*3, np.zeros(100)])
    pulse_neg = Pulse(data_neg)
    amplitude_neg = pulse_neg.find_amplitude()
    assert amplitude_neg > 0

    data_zeros = np.zeros(200)
    pulse_zeros = Pulse(data_zeros)
    amplitude_zeros = pulse_zeros.find_amplitude()
    assert amplitude_zeros == 1  # as per code

def test_normalize_deriv():
    data = np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.linspace(1, 0, 50), np.zeros(100)])
    pulse = Pulse(data)
    norm_deriv = pulse.normalize_deriv(window_length=11)
    # Main peak in derivative should be +1 or -1
    assert np.isclose(np.max(np.abs(norm_deriv)), 1, atol=1e-2)
    # Check only the central part of the flat region, far from edges
    assert np.allclose(norm_deriv[20:80], 0, atol=0.02)
    assert np.allclose(norm_deriv[-80:-20], 0, atol=0.02)

    data_zeros = np.zeros(200)
    pulse_zeros = Pulse(data_zeros)
    norm_deriv_zeros = pulse_zeros.normalize_deriv(window_length=11)
    assert np.allclose(norm_deriv_zeros, 0)

if __name__ == "__main__":
    test_pulse()
    test_find_rise_time()
    test_find_decay_time()
    test_find_peak_fwhm_time()
    test_calculate_area()
    test_normalize_pulse()
    test_count_peaks()
    test_l1_norm()
    test_find_amplitude()
    test_normalize_deriv()