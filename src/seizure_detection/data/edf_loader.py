import mne
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any

class EDFLoadError(Exception):
    """Raised when an EDF file cannot be loaded or fails basic validation."""
    pass

def load_edf(file_path: str) -> Tuple[np.ndarray, List[str], float]:
    """
    Loads an EDF file using MNE.
    
    Args:
        file_path (str): Path to the EDF file.
        
    Returns:
        Tuple containing:
            - signals (np.ndarray): Shape (n_channels, n_samples). Values are typically in Volts (as returned by MNE) 
                                    without arbitrary clipping or scaling.
            - ch_names (List[str]): List of channel names
            - sfreq (float): Sampling frequency in Hz
            
    Raises:
        EDFLoadError: If the file is not found, cannot be read, contains NaNs/Infs.
    """
    if not Path(file_path).exists():
        raise EDFLoadError(f"File not found: {file_path}")
        
    try:
        # Load raw data; preload=True loads data into memory
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        raise EDFLoadError(f"Failed to load EDF {file_path}: {str(e)}")
        
    signals = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']
    
    if np.isnan(signals).any() or np.isinf(signals).any():
        raise EDFLoadError(f"EDF {file_path} contains non-finite values (NaN or Inf).")
        
    if signals.size == 0:
        raise EDFLoadError(f"EDF {file_path} is empty.")
        
    if not np.isclose(sfreq, 256.0):
        raise EDFLoadError(f"EDF {file_path} has sampling rate {sfreq} Hz, but 256 Hz is explicitly required.")
        
    return signals, ch_names, sfreq

def get_edf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts metadata from an EDF without loading the full signal into memory.
    Useful for auditing.
    """
    if not Path(file_path).exists():
        return {"status": "error", "reason": "File not found"}
        
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        return {
            "status": "success",
            "channels": raw.ch_names,
            "n_channels": len(raw.ch_names),
            "sfreq": raw.info['sfreq'],
            "n_samples": raw.n_times,
            "duration_sec": raw.times[-1] if len(raw.times) > 0 else 0
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}
