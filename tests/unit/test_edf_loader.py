import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from seizure_detection.data.edf_loader import load_edf, get_edf_metadata, EDFLoadError

def test_load_edf_missing_file():
    with pytest.raises(EDFLoadError, match="File not found"):
        load_edf("fake_file.edf")

@patch("mne.io.read_raw_edf")
@patch("pathlib.Path.exists")
def test_load_edf_success(mock_exists, mock_read):
    mock_exists.return_value = True
    # Mock MNE Raw object
    mock_raw = MagicMock()
    mock_raw.get_data.return_value = np.zeros((18, 2560))
    mock_raw.ch_names = ["FP1-F7"] * 18
    mock_raw.info = {"sfreq": 256.0}
    mock_read.return_value = mock_raw
    
    signals, ch_names, sfreq = load_edf("dummy.edf")
    assert signals.shape == (18, 2560)
    assert sfreq == 256.0
    assert len(ch_names) == 18

@patch("mne.io.read_raw_edf")
@patch("pathlib.Path.exists")
def test_load_edf_nan_raises_error(mock_exists, mock_read):
    mock_exists.return_value = True
    mock_raw = MagicMock()
    bad_data = np.zeros((18, 2560))
    bad_data[0, 100] = np.nan
    mock_raw.get_data.return_value = bad_data
    mock_raw.info = {"sfreq": 256.0}
    mock_read.return_value = mock_raw
    
    with pytest.raises(EDFLoadError, match="contains non-finite values"):
        load_edf("dummy.edf")
