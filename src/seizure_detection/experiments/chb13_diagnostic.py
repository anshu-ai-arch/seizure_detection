import mne
import numpy as np
import logging

from typing import List

logger = logging.getLogger(__name__)

def investigate_chb13_duplicates(edf_path: str, chunk_size: int = 256000) -> dict:
    """
    Diagnostic utility to investigate duplicated T8-P8 channels (T8-P8-0, T8-P8-1)
    across the entire recording in bounded-memory chunks.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ch_names = raw.ch_names

        t8_p8_channels = [ch for ch in ch_names if 'T8-P8' in ch.upper()]

        logger.info(f"File: {edf_path}")
        logger.info(f"Found T8-P8 variants: {t8_p8_channels}")

        if len(t8_p8_channels) >= 2:
            idx0 = ch_names.index(t8_p8_channels[0])
            idx1 = ch_names.index(t8_p8_channels[1])

            total_samples = raw.n_times
            samples_checked = 0
            max_diff = 0.0
            are_identical = True
            has_non_finite = False

            # Process in chunks
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)

                sig0, _ = raw[idx0, start_idx:end_idx]
                sig1, _ = raw[idx1, start_idx:end_idx]

                sig0 = sig0.flatten()
                sig1 = sig1.flatten()

                # Check for non-finite values
                if not np.isfinite(sig0).all() or not np.isfinite(sig1).all():
                    has_non_finite = True

                if are_identical:
                    if not np.allclose(sig0, sig1, equal_nan=True):
                        are_identical = False

                diff = np.abs(sig0 - sig1)
                valid_diff = diff[np.isfinite(diff)]
                if len(valid_diff) > 0:
                    chunk_max_diff = np.max(valid_diff)
                    if chunk_max_diff > max_diff:
                        max_diff = float(chunk_max_diff)

                samples_checked += (end_idx - start_idx)

            logger.info(f"Comparison (Total samples checked: {samples_checked}):")
            logger.info(f" - Are completely identical: {are_identical}")
            logger.info(f" - Max absolute difference: {max_diff}")
            logger.info(f" - Has non-finite values: {has_non_finite}")

            return {
                "file": edf_path,
                "variants": t8_p8_channels,
                "identical": bool(are_identical),
                "max_diff": float(max_diff),
                "samples_checked": samples_checked,
                "has_non_finite": has_non_finite
            }
        else:
            logger.info("Not enough variants to compare.")
            return {"file": edf_path, "variants": t8_p8_channels}

    except Exception as e:
        logger.error(f"Failed to investigate {edf_path}: {e}")
        return None

def run_diagnostics_on_files(files: List[str]):
    for f in files:
        investigate_chb13_duplicates(f)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf-paths", type=str, nargs='+', required=True, help="Paths to the EDF files to diagnose")
    args = parser.parse_args()
    run_diagnostics_on_files(args.edf_paths)
