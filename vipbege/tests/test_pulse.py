import numpy as np

import pytest

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

if __name__ == "__main__":
    test_pulse()
    test_find_rise_time()
