import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from seizure_detection.experiments.kaggle_runner import run_pipeline

@pytest.fixture
def mock_dataset_dir(tmp_path):
    patient_dir = tmp_path / "chb01"
    patient_dir.mkdir()
    
    summary_content = """
Data Sampling Rate: 256 Hz

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""
    with open(patient_dir / "chb01-summary.txt", "w") as f:
        f.write(summary_content)
        
    (patient_dir / "chb01_01.edf").touch()
    (patient_dir / "chb01_03.edf").touch()
    (patient_dir / "chb01_unlisted.edf").touch()
    
    return tmp_path

@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_kaggle_runner_pipeline(mock_load_edf, mock_dataset_dir):
    # Mock successful load
    def side_effect(file_path):
        signals = np.zeros((18, 256 * 3600)) # 1 hour
        ch_names = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"
        ]
        return signals, ch_names, 256.0
        
    mock_load_edf.side_effect = side_effect
    
    out_dir = mock_dataset_dir / "out"
    
    run_pipeline(
        data_dir=str(mock_dataset_dir),
        out_dir=str(out_dir),
        config_path="configs/preprocessing.yaml",
        limit_recordings=None,
        dry_run=False
    )
    
    audit_file = out_dir / "audit_report.csv"
    assert audit_file.exists()
    
    with open(audit_file, "r") as f:
        content = f.read()
        
    assert "chb01_01.edf" in content
    assert "chb01_03.edf" in content
    assert "chb01_unlisted.edf" in content
    assert "False" in content # At least one file is is_listed=False (chb01_unlisted)

@patch("seizure_detection.experiments.kaggle_runner.get_edf_metadata")
def test_kaggle_runner_dry_run(mock_get_metadata, mock_dataset_dir):
    mock_get_metadata.return_value = {
        "status": "success",
        "channels": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"
        ],
        "n_channels": 18,
        "sfreq": 256.0,
        "n_samples": 256 * 3600,
        "duration_sec": 3600.0
    }
    
    out_dir = mock_dataset_dir / "out"
    
    run_pipeline(
        data_dir=str(mock_dataset_dir),
        out_dir=str(out_dir),
        config_path="configs/preprocessing.yaml",
        limit_recordings=1,
        dry_run=True
    )
    
    audit_file = out_dir / "audit_report.csv"
    assert audit_file.exists()
