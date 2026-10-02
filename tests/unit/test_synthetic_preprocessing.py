import pytest
import numpy as np
import yaml
from pathlib import Path

from seizure_detection.preprocessing.channel_mapping import map_channels, MissingChannelError
from seizure_detection.preprocessing.dwt import apply_dwt, process_multichannel_dwt
from seizure_detection.windowing.window_generator import generate_windows

def test_channel_mapping_success():
    expected = ["FP1-F7", "F7-T7"]
    raw_names = ["X1", "FP1-F7", "F7-T7", "Z2"]
    raw_signals = np.random.randn(4, 1000)
    
    mapped = map_channels(raw_signals, raw_names, expected)
    assert mapped.shape == (2, 1000)
    assert np.allclose(mapped[0], raw_signals[1])
    
def test_channel_mapping_missing_raises_error():
    expected = ["FP1-F7", "MISSING-CH"]
    raw_names = ["FP1-F7", "F7-T7"]
    raw_signals = np.random.randn(2, 1000)
    
    with pytest.raises(MissingChannelError):
        map_channels(raw_signals, raw_names, expected)

def test_dwt_shape_and_filtering():
    # Synthetic signal 10 seconds at 256 Hz
    fs = 256
    t = np.linspace(0, 10, 10*fs)
    # Add 2 Hz (should be kept) and 100 Hz (should be removed by cD1/cD2 zeroing)
    signal = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 100 * t)
    
    filtered = apply_dwt(signal, wavelet='db4', level=5)
    
    assert len(filtered) == len(signal)
    
    # Check that high frequencies are reduced
    fft_orig = np.abs(np.fft.rfft(signal))
    fft_filt = np.abs(np.fft.rfft(filtered))
    
    # 100 Hz index approx 100 / (128) * len(rfft) -> rough check
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    idx_100hz = np.argmin(np.abs(freqs - 100))
    
    assert fft_filt[idx_100hz] < fft_orig[idx_100hz] * 0.1 # Should be heavily attenuated

def test_multichannel_dwt():
    signals = np.random.randn(18, 2560)
    filtered = process_multichannel_dwt(signals)
    assert filtered.shape == (18, 2560)

def test_window_generation_non_seizure_no_overlap():
    # 10 seconds of data at 256 Hz
    signals = np.random.randn(18, 2560)
    windows = generate_windows(signals, is_seizure=False, sampling_rate=256, window_sec=4, overlap_non_seizure=0.0)
    
    # Should get exactly 2 windows (0-4s, 4-8s). The remaining 2s is dropped (unless boundary rules are defined).
    assert len(windows) == 2
    assert windows[0].shape == (18, 1024)
    
def test_window_generation_seizure_overlap():
    # 10 seconds of data at 256 Hz
    signals = np.random.randn(18, 2560)
    # 4s window, 50% overlap means stride is 2s (512 samples)
    # Windows start at: 0s, 2s, 4s, 6s -> total 4 windows
    windows = generate_windows(signals, is_seizure=True, sampling_rate=256, window_sec=4, overlap_seizure=0.5)
    
    assert len(windows) == 4
    assert windows[0].shape == (18, 1024)

def test_window_generation_nan_raises_error():
    signals = np.random.randn(18, 2560)
    signals[0, 500] = np.nan
    with pytest.raises(ValueError, match="NaN found"):
        generate_windows(signals, is_seizure=False)
