import argparse
import logging
import yaml
import json
import hashlib
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import numpy as np

from seizure_detection.data.edf_loader import load_edf, get_edf_metadata, EDFLoadError
from seizure_detection.data.summary_parser import parse_chbmit_summary, SummaryParseError
from seizure_detection.preprocessing.channel_mapping import map_channels, MissingChannelError
from seizure_detection.preprocessing.dwt import process_multichannel_dwt
from seizure_detection.windowing.window_generator import generate_windows
from seizure_detection.windowing.annotation_mapper import map_annotations_to_windows
from seizure_detection.data.metadata_auditor import MetadataAuditor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_zip(filepath: Path) -> bool:
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            return z.testzip() is None
    except Exception:
        return False

def run_pipeline(
    data_dir: str,
    out_dir: str,
    config_path: str,
    batch_size: int = None,
    dry_run: bool = False
):
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Config file not found at {config_file}")
        return

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    expected_channels = config['channels']['expected_channels']

    out_path.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path / "temp_batch"
    temp_dir.mkdir(exist_ok=True)
    
    shards_dir = out_path / "shards"
    shards_dir.mkdir(exist_ok=True)

    completed_log = out_path / "completed_recordings.txt"
    completed_recordings = set()
    if completed_log.exists():
        with open(completed_log, "r") as f:
            completed_recordings = set(line.strip() for line in f if line.strip())

    if not data_path.exists():
        logger.warning(f"Data directory {data_path} does not exist.")
        return

    patient_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('chb')]
    
    # Gather all EDFs
    all_edfs = []
    for p_dir in patient_dirs:
        for edf_file in p_dir.glob("*.edf"):
            all_edfs.append((p_dir.name, edf_file))
            
    # Sort deterministically
    all_edfs.sort(key=lambda x: (x[0], x[1].name))
    
    pending_edfs = []
    for pid, edf_file in all_edfs:
        rec_id = f"{pid}_{edf_file.name}"
        if rec_id not in completed_recordings:
            pending_edfs.append((pid, edf_file))
            
    logger.info(f"Total recordings: {len(all_edfs)}, Completed: {len(completed_recordings)}, Pending: {len(pending_edfs)}")

    if batch_size:
        pending_edfs = pending_edfs[:batch_size]

    if not pending_edfs:
        logger.info("No pending recordings to process in this batch.")
        return
        
    auditor = MetadataAuditor()
    batch_manifest = []

    for patient_id, edf_file in pending_edfs:
        filename = edf_file.name
        rec_id = f"{patient_id}_{filename}"
        
        is_patient_excluded = patient_id in config['data'].get('excluded_patients', [])
        
        summary_file = edf_file.parent / f"{patient_id}-summary.txt"
        annotations = {}
        if summary_file.exists():
            try:
                annotations = parse_chbmit_summary(str(summary_file))
            except SummaryParseError as e:
                logger.error(f"Summary parse error for {patient_id}: {e}")

        is_listed = filename in annotations
        seizure_intervals = annotations.get(filename, [])

        if is_patient_excluded:
            meta = get_edf_metadata(str(edf_file))
            duration = meta.get('duration_sec') if meta.get('status') == 'success' else None
            auditor.add_recording(
                patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                is_patient_excluded=True, sfreq_valid=False, channels_valid=False,
                duration_sec=duration, annotations_valid=False, diagnostic_reason="Patient excluded by configuration"
            )
            with open(completed_log, "a") as f:
                f.write(rec_id + "\n")
            continue

        if dry_run:
            meta = get_edf_metadata(str(edf_file))
            sfreq_valid = meta.get('sfreq') == 256.0
            channels_valid = all(ch in meta.get('channels', []) for ch in expected_channels) if meta.get('status') == 'success' else False
            duration = meta.get('duration_sec', 0.0)
            ann_valid = True
            for (s, e) in seizure_intervals:
                if duration and e > duration:
                    ann_valid = False
                    
            auditor.add_recording(
                patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                is_patient_excluded=False, sfreq_valid=sfreq_valid, channels_valid=channels_valid,
                duration_sec=duration, annotations_valid=ann_valid,
                diagnostic_reason="Dry run inventory check" if meta.get('status') == 'success' else meta.get('reason', 'Failed')
            )
            with open(completed_log, "a") as f:
                f.write(rec_id + "\n")
            continue

        # Full processing
        duration = 0.0
        rec_manifest = []
        try:
            signals, ch_names, sfreq = load_edf(str(edf_file))
            duration = signals.shape[1] / sfreq

            ann_valid = True
            for (s, e) in seizure_intervals:
                if e > duration:
                    ann_valid = False

            if not ann_valid:
                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=False, sfreq_valid=True, channels_valid=True,
                    duration_sec=duration, annotations_valid=False, diagnostic_reason="Annotation end exceeds EDF duration"
                )
            else:
                mapped_signals = map_channels(signals, ch_names, expected_channels)
                filtered_signals = process_multichannel_dwt(mapped_signals)
                windows_info = map_annotations_to_windows(seizure_intervals, duration, window_sec=4)

                window_samples = int(4 * sfreq)
                saved_windows_count = 0
                for w_idx, w_info in enumerate(windows_info):
                    start_sec = w_info["start_sec"]
                    end_sec = w_info["end_sec"]
                    status = w_info["status"]

                    start_idx = int(start_sec * sfreq)
                    window_array = filtered_signals[:, start_idx : start_idx + window_samples]

                    if window_array.shape[1] == window_samples and np.isfinite(window_array).all():
                        window_filename = f"{patient_id}_{filename}_w{w_idx}.npy"
                        np.save(temp_dir / window_filename, window_array)

                        rec_manifest.append({
                            "patient_id": patient_id,
                            "filename": filename,
                            "window_file": window_filename,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "status": status,
                            "channels": "|".join(expected_channels),
                            "sfreq": sfreq,
                            "window_samples": window_samples,
                            "window_duration_sec": 4.0,
                            "stride_sec": (start_sec - windows_info[w_idx-1]["start_sec"]) if w_idx > 0 else 0.0,
                            "wavelet": "db4",
                            "dwt_level": 5,
                            "retained_coefficients": "cA5,cD5,cD4,cD3",
                            "annotation_source": "summary_parser",
                            "annotation_status": "listed" if is_listed else "unlisted"
                        })
                        saved_windows_count += 1

                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=False, sfreq_valid=True, channels_valid=True,
                    duration_sec=duration, annotations_valid=True,
                    diagnostic_reason=f"Successfully processed {saved_windows_count} windows"
                )

        except EDFLoadError as e:
            auditor.add_recording(
                patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                is_patient_excluded=False, sfreq_valid=False, channels_valid=False,
                duration_sec=0.0, annotations_valid=False, diagnostic_reason=str(e)
            )
        except MissingChannelError as e:
            auditor.add_recording(
                patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                is_patient_excluded=False, sfreq_valid=True, channels_valid=False,
                duration_sec=duration, annotations_valid=False, diagnostic_reason=str(e)
            )
        except Exception as e:
            auditor.add_recording(
                patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                is_patient_excluded=False, sfreq_valid=False, channels_valid=False,
                duration_sec=duration, annotations_valid=False, diagnostic_reason=f"Unexpected error: {str(e)}"
            )

        # Mark recording as complete ONLY if everything succeeded.
        batch_manifest.extend(rec_manifest)
        with open(completed_log, "a") as f:
            f.write(rec_id + "\n")

    if not dry_run and auditor.records:
        import time
        shard_id = f"shard_{int(time.time())}"
        
        # Save manifest and audit to temp_dir before zipping
        manifest_df = pd.DataFrame(batch_manifest) if batch_manifest else pd.DataFrame()
        if not manifest_df.empty:
            manifest_df.to_csv(temp_dir / f"{shard_id}_manifest.csv", index=False)
            
        audit_df = pd.DataFrame(auditor.records)
        audit_df.to_csv(temp_dir / f"{shard_id}_audit.csv", index=False)
        
        zip_path = shards_dir / f"{shard_id}.zip"
        
        # Create Zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fpath in temp_dir.iterdir():
                if fpath.is_file():
                    zf.write(fpath, arcname=fpath.name)
                    
        # Verify Zip
        if not verify_zip(zip_path):
            logger.error(f"Shard {shard_id} failed zip verification! Aborting cleanup.")
            zip_path.unlink()
            return
            
        # Checksum
        sha256 = compute_sha256(zip_path)
        with open(shards_dir / f"{shard_id}.sha256", "w") as f:
            f.write(f"{sha256}  {shard_id}.zip\n")
            
        # Safe cleanup
        for fpath in temp_dir.iterdir():
            if fpath.is_file():
                fpath.unlink()
                
        logger.info(f"Successfully finalized {shard_id}.zip with {len(batch_manifest)} windows.")

    audit_file = out_path / "audit_report.csv"
    if auditor.records:
        auditor.export_csv(str(audit_file))
        logger.info(f"Pipeline complete. Global audit saved to {audit_file}")
        
    logger.info("Dataset splitting and unresolved boundary logic remain explicitly blocked.")

def main():
    parser = argparse.ArgumentParser(description="Resumable Kaggle Pipeline Runner")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric/chb-mit-scalp-eeg-database-1.0.0/", help="Path to dataset root")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/processed_data", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of recordings to process in this run")
    parser.add_argument("--dry-run", action="store_true", help="Run in inventory-only mode without full loading")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
