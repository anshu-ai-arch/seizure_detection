import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import json
import zipfile
import hashlib
import pandas as pd

from seizure_detection.experiments.kaggle_runner import run_pipeline
from seizure_detection.experiments.recovery_utils import inventory_shards, compute_sha256
from seizure_detection.data.edf_loader import EDFLoadError

@pytest.fixture
def mock_resume_env(tmp_path):
    dataset_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    dataset_dir.mkdir()
    out_dir.mkdir()
    
    patient_dir = dataset_dir / "chb01"
    patient_dir.mkdir()
    
    (patient_dir / "chb01_01.edf").touch()
    (patient_dir / "chb01_02.edf").touch()
    
    with open(patient_dir / "chb01-summary.txt", "w") as f:
        f.write("Data Sampling Rate: 256 Hz\nFile Name: chb01_01.edf\nNumber of Seizures in File: 0\nFile Name: chb01_02.edf\nNumber of Seizures in File: 0\n")
        
    return dataset_dir, out_dir

def _mock_eeg_signals():
    signals = np.zeros((18, 256 * 10)) 
    ch_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ"
    ]
    return signals, ch_names, 256.0

# 1. A failed EDF is not marked completed
@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_failed_edf_not_marked_completed(mock_load_edf, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    mock_load_edf.side_effect = EDFLoadError("Corrupt EDF header")

    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=1, dry_run=False)

    completed_log = out_dir / "completed_recordings.txt"
    if completed_log.exists():
        with open(completed_log, "r") as f:
            completed = [line.strip() for line in f if line.strip()]
        assert "chb01_chb01_01.edf" not in completed
    else:
        assert not completed_log.exists()

    audit_file = out_dir / "audit_report.csv"
    assert audit_file.exists()
    audit_df = pd.read_csv(audit_file)
    assert "chb01_01.edf" in audit_df["filename"].values
    rec_row = audit_df[audit_df["filename"] == "chb01_01.edf"].iloc[0]
    assert rec_row["status"] == "EXCLUDED_SFREQ" or "Corrupt EDF header" in str(rec_row["reason"])

# 2. A successful EDF is marked completed only after archive verification
@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_successful_edf_marked_completed_only_after_archive_verification(mock_load_edf, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    mock_load_edf.side_effect = lambda f: _mock_eeg_signals()

    # Part A: Normal successful run marks completed and creates valid archive
    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=1, dry_run=False)

    completed_log = out_dir / "completed_recordings.txt"
    assert completed_log.exists()
    with open(completed_log, "r") as f:
        completed = [line.strip() for line in f if line.strip()]
    assert "chb01_chb01_01.edf" in completed

    shards = list((out_dir / "shards").glob("*.zip"))
    assert len(shards) == 1
    sha_file = out_dir / "shards" / f"{shards[0].stem}.sha256"
    assert sha_file.exists()
    with open(sha_file, "r") as f:
        expected_sha = f.read().split()[0]
    assert compute_sha256(shards[0]) == expected_sha

    # Part B: If zip verification fails, recording is NOT marked completed and corrupt zip unlinked
    out_dir_fail = out_dir.parent / "out_fail"
    out_dir_fail.mkdir()
    with patch("seizure_detection.experiments.kaggle_runner.verify_zip", return_value=False):
        run_pipeline(str(dataset_dir), str(out_dir_fail), "configs/preprocessing.yaml", batch_size=1, dry_run=False)

    completed_fail = out_dir_fail / "completed_recordings.txt"
    assert not completed_fail.exists()
    assert len(list((out_dir_fail / "shards").glob("*.zip"))) == 0

# 3. Resume retries failed recordings but skips genuinely completed recordings
@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_resume_retries_failed_recordings_and_skips_completed(mock_load_edf, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env

    # Run 1: chb01_01 succeeds, chb01_02 fails
    def first_run_loader(filepath):
        if "chb01_01" in str(filepath):
            return _mock_eeg_signals()
        raise EDFLoadError("Transient read error")
    mock_load_edf.side_effect = first_run_loader

    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=2, dry_run=False)

    completed_log = out_dir / "completed_recordings.txt"
    assert completed_log.exists()
    with open(completed_log, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1
    assert "chb01_chb01_01.edf" in lines[0]
    assert "chb01_chb01_02.edf" not in lines

    # Run 2: chb01_02 is fixed and succeeds; chb01_01 must NOT be called again
    loaded_files = []
    def second_run_loader(filepath):
        loaded_files.append(str(filepath))
        return _mock_eeg_signals()
    mock_load_edf.side_effect = second_run_loader

    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=None, dry_run=False)

    # Verify chb01_01 was skipped
    assert not any("chb01_01.edf" in f for f in loaded_files)
    assert any("chb01_02.edf" in f for f in loaded_files)

    with open(completed_log, "r") as f:
        updated_lines = [l.strip() for l in f if l.strip()]
    assert len(updated_lines) == 2
    assert "chb01_chb01_02.edf" in updated_lines[1]

# 4. Corrupt ZIPs and checksum mismatches are detected
def test_corrupt_zips_and_checksum_mismatches_detected(mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)

    # 1. Valid zip with matching checksum
    valid_zip = shards_dir / "shard_valid.zip"
    with zipfile.ZipFile(valid_zip, 'w') as zf:
        zf.writestr("shard_valid_manifest.csv", "patient_id,filename,window_file\nchb01,chb01_01.edf,w0.npy\n")
    sha_valid = hashlib.sha256(valid_zip.read_bytes()).hexdigest()
    with open(shards_dir / "shard_valid.sha256", "w") as f:
        f.write(f"{sha_valid}  shard_valid.zip\n")

    # 2. Corrupt zip content
    corrupt_zip = shards_dir / "shard_corrupt.zip"
    corrupt_zip.write_text("not a real zip content")
    sha_corrupt = hashlib.sha256(corrupt_zip.read_bytes()).hexdigest()
    with open(shards_dir / "shard_corrupt.sha256", "w") as f:
        f.write(f"{sha_corrupt}  shard_corrupt.zip\n")

    # 3. Valid zip with wrong checksum in sha256 file
    mismatch_zip = shards_dir / "shard_mismatch.zip"
    with zipfile.ZipFile(mismatch_zip, 'w') as zf:
        zf.writestr("test.txt", "data")
    with open(shards_dir / "shard_mismatch.sha256", "w") as f:
        f.write("0000000000000000000000000000000000000000000000000000000000000000  shard_mismatch.zip\n")

    # 4. Valid zip with missing sha256 file
    nosha_zip = shards_dir / "shard_nosha.zip"
    with zipfile.ZipFile(nosha_zip, 'w') as zf:
        zf.writestr("test.txt", "data")

    report = inventory_shards(str(out_dir))

    assert len(report["valid_shards"]) == 1
    assert report["valid_shards"][0]["shard"] == "shard_valid.zip"
    assert report["valid_shards"][0]["checksum_match"] is True

    invalid_names = {s["shard"] for s in report["invalid_shards"]}
    assert "shard_corrupt.zip" in invalid_names
    assert "shard_mismatch.zip" in invalid_names
    assert "shard_nosha.zip" in invalid_names

# 5. Dry runs do not modify real-run checkpoints
@patch("seizure_detection.experiments.kaggle_runner.get_edf_metadata")
def test_dry_runs_do_not_modify_checkpoints(mock_get_metadata, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    mock_get_metadata.return_value = {
        "status": "success",
        "sfreq": 256.0,
        "channels": [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"
        ],
        "duration_sec": 3600.0
    }

    # Pre-populate checkpoint
    completed_log = out_dir / "completed_recordings.txt"
    completed_log.write_text("chb01_prior_checkpoint.edf\n")

    # Add an excluded patient to ensure excluded logic does not modify checkpoint either
    excluded_dir = dataset_dir / "chb06"
    excluded_dir.mkdir()
    (excluded_dir / "chb06_01.edf").touch()

    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=None, dry_run=True)

    # Checkpoint must be strictly unmodified
    assert completed_log.read_text() == "chb01_prior_checkpoint.edf\n"

    # No shards created
    shards_dir = out_dir / "shards"
    assert not shards_dir.exists() or len(list(shards_dir.glob("*.zip"))) == 0

    # Audit report exists and contains dry run records
    audit_file = out_dir / "audit_report.csv"
    assert audit_file.exists()

# 6. Recovery report counts match the actual shard manifests
@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_recovery_report_counts_match_actual_shard_manifests(mock_load_edf, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    mock_load_edf.side_effect = lambda f: _mock_eeg_signals()

    # Process 1 file
    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=1, dry_run=False)

    # Inspect shard directly to count expected windows
    shards = list((out_dir / "shards").glob("*.zip"))
    assert len(shards) == 1
    with zipfile.ZipFile(shards[0], 'r') as zf:
        manifest_names = [n for n in zf.namelist() if n.endswith("_manifest.csv")]
        with zf.open(manifest_names[0]) as mf:
            df = pd.read_csv(mf)
            expected_windows = len(df)

    assert expected_windows > 0

    # Run inventory with data_dir provided
    report = inventory_shards(str(out_dir), data_dir=str(dataset_dir))

    assert report["manifest_coverage_windows"] == expected_windows
    assert report["completed_recordings_count"] == 1
    assert "chb01_chb01_02.edf" in report["incomplete_or_missing"]
    assert "chb01_chb01_01.edf" not in report["incomplete_or_missing"]

    # Run inventory without data_dir: should report unavailable rather than misleading []
    report_no_data = inventory_shards(str(out_dir))
    assert report_no_data["manifest_coverage_windows"] == expected_windows
    assert report_no_data["completed_recordings_count"] == 1
    assert report_no_data["incomplete_or_missing"] == "unavailable (data_dir not provided)"

# 7. Shard creation excludes stale files
@patch("seizure_detection.experiments.kaggle_runner.load_edf")
def test_shard_creation_excludes_stale_files(mock_load_edf, mock_resume_env):
    dataset_dir, out_dir = mock_resume_env
    mock_load_edf.side_effect = lambda f: _mock_eeg_signals()

    # Pre-seed temp_batch with stale file
    temp_dir = out_dir / "temp_batch"
    temp_dir.mkdir(parents=True)
    stale_file = temp_dir / "stale_leftover_w999.npy"
    stale_file.write_text("stale data")

    run_pipeline(str(dataset_dir), str(out_dir), "configs/preprocessing.yaml", batch_size=1, dry_run=False)

    shards = list((out_dir / "shards").glob("*.zip"))
    assert len(shards) == 1
    with zipfile.ZipFile(shards[0], 'r') as zf:
        names = zf.namelist()
        assert "stale_leftover_w999.npy" not in names
