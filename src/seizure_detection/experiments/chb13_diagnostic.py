import mne
import numpy as np
import logging

logger = logging.getLogger(__name__)

def investigate_chb13_duplicates(edf_path: str, n_samples: int = 2560):
    """
    Diagnostic utility to investigate duplicated T8-P8 channels (T8-P8-0, T8-P8-1)
    without loading the full hour-long array into memory.
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
            
            # Extract only a small memory-conscious sample (e.g., 10 seconds)
            sig0, _ = raw[idx0, :n_samples]
            sig1, _ = raw[idx1, :n_samples]
            
            sig0 = sig0.flatten()
            sig1 = sig1.flatten()
            
            are_identical = np.allclose(sig0, sig1, equal_nan=True)
            max_diff = np.max(np.abs(sig0 - sig1)) if not are_identical else 0.0
            
            logger.info(f"Comparison ({n_samples} samples):")
            logger.info(f" - Are identical: {are_identical}")
            logger.info(f" - Max absolute difference: {max_diff}")
            
            return {
                "file": edf_path,
                "variants": t8_p8_channels,
                "identical": bool(are_identical),
                "max_diff": float(max_diff)
            }
        else:
            logger.info("Not enough variants to compare.")
            return {"file": edf_path, "variants": t8_p8_channels}
            
    except Exception as e:
        logger.error(f"Failed to investigate {edf_path}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf-path", type=str, required=True, help="Path to the EDF file to diagnose")
    args = parser.parse_args()
    investigate_chb13_duplicates(args.edf_path)
