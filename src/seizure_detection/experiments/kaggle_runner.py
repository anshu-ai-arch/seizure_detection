import argparse
import logging
import yaml
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

def run_pipeline(
    data_dir: str,
    out_dir: str,
    config_path: str,
    limit_recordings: int = None,
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

    auditor = MetadataAuditor()
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.warning(f"Data directory {data_path} does not exist.")
        auditor.export_csv(str(out_path / "audit_report.csv"))
        return

    # Find all patients (directories like chb01, chb02, etc.)
    patient_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('chb')]

    recordings_processed = 0
    window_manifest = []

    for p_dir in patient_dirs:
        patient_id = p_dir.name
        is_patient_excluded = patient_id in config['data'].get('excluded_patients', [])

        summary_file = p_dir / f"{patient_id}-summary.txt"
        annotations = {}
        if summary_file.exists():
            try:
                annotations = parse_chbmit_summary(str(summary_file))
            except SummaryParseError as e:
                logger.error(f"Summary parse error for {patient_id}: {e}")
                # We still process EDFs but they will be marked as unlisted/unresolved.

        edf_files = list(p_dir.glob("*.edf"))
        patient_recordings_processed = 0

        for edf_file in edf_files:
            if limit_recordings and patient_recordings_processed >= limit_recordings:
                break

            filename = edf_file.name
            is_listed = filename in annotations
            seizure_intervals = annotations.get(filename, [])

            # Dry run / fast metadata extraction
            if dry_run:
                meta = get_edf_metadata(str(edf_file))
                sfreq_valid = meta.get('sfreq') == 256.0
                channels_valid = all(ch in meta.get('channels', []) for ch in expected_channels) if meta.get('status') == 'success' else False
                duration = meta.get('duration_sec', 0.0)
                ann_valid = True

                # Check annotation duration valid
                for (s, e) in seizure_intervals:
                    if e > duration:
                        ann_valid = False

                auditor.add_recording(
                    patient_id=patient_id,
                    filename=filename,
                    is_listed_in_summary=is_listed,
                    is_patient_excluded=is_patient_excluded,
                    sfreq_valid=sfreq_valid,
                    channels_valid=channels_valid,
                    duration_sec=duration,
                    annotations_valid=ann_valid,
                    diagnostic_reason="Dry run inventory check" if meta.get('status') == 'success' else meta.get('reason', 'Failed')
                )
                recordings_processed += 1
                continue

            # Full processing
            duration = 0.0
            try:
                signals, ch_names, sfreq = load_edf(str(edf_file))
                duration = signals.shape[1] / sfreq

                # Validate annotations fit in duration
                ann_valid = True
                for (s, e) in seizure_intervals:
                    if e > duration:
                        ann_valid = False

                if not ann_valid:
                    auditor.add_recording(
                        patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                        is_patient_excluded=is_patient_excluded, sfreq_valid=True, channels_valid=True,
                        duration_sec=duration, annotations_valid=False, diagnostic_reason="Annotation end exceeds EDF duration"
                    )
                    continue

                mapped_signals = map_channels(signals, ch_names, expected_channels)

                # We can perform DWT safely
                filtered_signals = process_multichannel_dwt(mapped_signals)

                # Window generation
                windows_info = map_annotations_to_windows(seizure_intervals, duration, window_sec=4)

                # Extract the physical windows
                window_samples = int(4 * sfreq)
                saved_windows_count = 0
                for w_idx, w_info in enumerate(windows_info):
                    start_sec = w_info["start_sec"]
                    end_sec = w_info["end_sec"]
                    status = w_info["status"]

                    start_idx = int(start_sec * sfreq)

                    window_array = filtered_signals[:, start_idx : start_idx + window_samples]

                    # Sanity check
                    if window_array.shape[1] == window_samples and np.isfinite(window_array).all():
                        window_filename = f"{patient_id}_{filename}_w{w_idx}.npy"
                        np.save(out_path / window_filename, window_array)

                        window_manifest.append({
                            "patient_id": patient_id,
                            "filename": filename,
                            "window_file": window_filename,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "status": status,
                            "channels": "|".join(expected_channels),
                            "sfreq": sfreq,
                            "dwt_config": "db4_level5_cA5_cD5_cD4_cD3"
                        })
                        saved_windows_count += 1

                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=is_patient_excluded, sfreq_valid=True, channels_valid=True,
                    duration_sec=duration, annotations_valid=True,
                    diagnostic_reason=f"Successfully processed {saved_windows_count} windows"
                )

            except EDFLoadError as e:
                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=is_patient_excluded, sfreq_valid=False, channels_valid=False,
                    duration_sec=0.0, annotations_valid=False, diagnostic_reason=str(e)
                )
            except MissingChannelError as e:
                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=is_patient_excluded, sfreq_valid=True, channels_valid=False,
                    duration_sec=duration, annotations_valid=False, diagnostic_reason=str(e)
                )
            except Exception as e:
                auditor.add_recording(
                    patient_id=patient_id, filename=filename, is_listed_in_summary=is_listed,
                    is_patient_excluded=is_patient_excluded, sfreq_valid=False, channels_valid=False,
                    duration_sec=duration, annotations_valid=False, diagnostic_reason=f"Unexpected error: {str(e)}"
                )

            patient_recordings_processed += 1
            recordings_processed += 1

    audit_file = out_path / "audit_report.csv"
    auditor.export_csv(str(audit_file))

    if window_manifest:
        import pandas as pd
        manifest_df = pd.DataFrame(window_manifest)
        manifest_file = out_path / "window_manifest.csv"
        manifest_df.to_csv(manifest_file, index=False)
        logger.info(f"Saved window manifest to {manifest_file} with {len(window_manifest)} windows.")

    logger.info(f"Pipeline complete. Audit saved to {audit_file}")
    logger.info("Dataset splitting and unresolved boundary logic remain explicitly blocked.")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Pipeline Runner")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric/chb-mit-scalp-eeg-database-1.0.0/", help="Path to CHB-MIT dataset root")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/processed_data", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml", help="Path to config file")
    parser.add_argument("--limit-recordings", type=int, default=None, help="Limit number of processed recordings per patient")
    parser.add_argument("--dry-run", action="store_true", help="Run in inventory-only mode without full loading")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        config_path=args.config,
        limit_recordings=args.limit_recordings,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
