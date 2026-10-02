import numpy as np
from typing import List, Dict, Any, Tuple

class UnresolvedBoundaryError(Exception):
    """Raised when a window crosses a seizure boundary and labeling logic is unapproved."""
    pass

def generate_windows(signals: np.ndarray, 
                     is_seizure: bool,
                     sampling_rate: int = 256,
                     window_sec: int = 4,
                     overlap_seizure: float = 0.5,
                     overlap_non_seizure: float = 0.0) -> List[np.ndarray]:
    """
    Slices a continuous segment into windows based on verified overlap rules.
    This function assumes the passed segment is entirely seizure or entirely non-seizure.
    Windows overlapping a boundary must be explicitly avoided or handled via an approved policy.
    
    Args:
        signals (np.ndarray): Shape (channels, samples)
        is_seizure (bool): True if the segment is a seizure event, False otherwise.
        sampling_rate (int): Hz
        window_sec (int): Duration of window in seconds
        
    Returns:
        List[np.ndarray]: List of window tensors of shape (channels, window_sec * sampling_rate)
    """
    window_samples = window_sec * sampling_rate
    overlap_ratio = overlap_seizure if is_seizure else overlap_non_seizure
    stride_samples = int(window_samples * (1 - overlap_ratio))
    
    num_samples = signals.shape[1]
    windows = []
    
    start_idx = 0
    while start_idx + window_samples <= num_samples:
        window = signals[:, start_idx : start_idx + window_samples]
        
        # Check for NaNs and block silent dropping (requires approval)
        if np.isnan(window).any():
            raise ValueError("NaN found in window. Explicit approval needed for dropping/imputing windows.")
            
        windows.append(window)
        start_idx += stride_samples
        
    return windows
