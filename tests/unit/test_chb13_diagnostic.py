import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from seizure_detection.experiments.chb13_diagnostic import investigate_chb13_duplicates

@pytest.fixture
def mock_mne_raw():
    raw = MagicMock()
    raw.ch_names = ["FP1-F7", "T8-P8-0", "T8-P8-1"]
    raw.n_times = 5000
    return raw

@patch("seizure_detection.experiments.chb13_diagnostic.mne.io.read_raw_edf")
def test_diagnostic_identical_channels(mock_read, mock_mne_raw):
    mock_read.return_value = mock_mne_raw

    # Mock __getitem__ to return identical signals for any indices
    def get_item(indices):
        idx = indices[0]
        slice_obj = indices[1]

        # create synthetic array of length matching slice
        start = slice_obj.start or 0
        stop = min(slice_obj.stop or mock_mne_raw.n_times, mock_mne_raw.n_times)

        sig = np.ones((1, stop - start)) * 0.5
        return sig, None

    mock_mne_raw.__getitem__.side_effect = get_item

    result = investigate_chb13_duplicates("dummy.edf", chunk_size=2000)

    assert result["identical"] is True
    assert result["max_diff"] == 0.0
    assert result["samples_checked"] == 5000
    assert result["has_non_finite"] is False

@patch("seizure_detection.experiments.chb13_diagnostic.mne.io.read_raw_edf")
def test_diagnostic_different_channels(mock_read, mock_mne_raw):
    mock_read.return_value = mock_mne_raw

    def get_item(indices):
        idx = indices[0]
        slice_obj = indices[1]
        start = slice_obj.start or 0
        stop = min(slice_obj.stop or mock_mne_raw.n_times, mock_mne_raw.n_times)

        # Make T8-P8-1 slightly different
        if idx == 1:
            sig = np.ones((1, stop - start)) * 0.5
        else:
            sig = np.ones((1, stop - start)) * 0.7
        return sig, None

    mock_mne_raw.__getitem__.side_effect = get_item

    result = investigate_chb13_duplicates("dummy.edf", chunk_size=2000)

    assert result["identical"] is False
    assert np.isclose(result["max_diff"], 0.2)
    assert result["samples_checked"] == 5000

@patch("seizure_detection.experiments.chb13_diagnostic.mne.io.read_raw_edf")
def test_diagnostic_non_finite_values(mock_read, mock_mne_raw):
    mock_read.return_value = mock_mne_raw

    def get_item(indices):
        idx = indices[0]
        slice_obj = indices[1]
        start = slice_obj.start or 0
        stop = min(slice_obj.stop or mock_mne_raw.n_times, mock_mne_raw.n_times)

        sig = np.ones((1, stop - start)) * 0.5
        # Inject NaN
        if start <= 1000 < stop:
            sig[0, 1000 - start] = np.nan

        return sig, None

    mock_mne_raw.__getitem__.side_effect = get_item

    result = investigate_chb13_duplicates("dummy.edf", chunk_size=2000)

    assert result["has_non_finite"] is True
    # np.allclose with equal_nan=True handles NaNs in same positions
    assert result["identical"] is True
