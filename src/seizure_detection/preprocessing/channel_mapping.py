import numpy as np
from typing import List, Optional

class MissingChannelError(Exception):
    """Raised when a required channel is missing from the EDF."""
    pass

def map_channels(raw_signals: np.ndarray, 
                 raw_channel_names: List[str], 
                 expected_channels: List[str]) -> np.ndarray:
    """
    Extracts and reorders raw signals to match the exact expected_channels list.
    
    Args:
        raw_signals (np.ndarray): Shape (num_raw_channels, num_samples)
        raw_channel_names (List[str]): Names corresponding to the rows of raw_signals
        expected_channels (List[str]): The exact 18 channels required by the paper
        
    Returns:
        np.ndarray: Shape (len(expected_channels), num_samples)
        
    Raises:
        MissingChannelError: If an expected channel is not present and no imputation rule is approved.
    """
    # Check for duplicates in expected channels
    if len(set(expected_channels)) != len(expected_channels):
        raise ValueError("Duplicate channels found in expected_channels list.")
        
    # Check for duplicates in raw channel names
    if len(set(raw_channel_names)) != len(raw_channel_names):
        raise ValueError("Duplicate channel names found in raw EDF recording.")
        
    selected_signals = []
    
    raw_names_upper = [ch.upper() for ch in raw_channel_names]
    
    for ch in expected_channels:
        ch_upper = ch.upper()
        if ch_upper not in raw_names_upper:
            raise MissingChannelError(f"Required channel '{ch}' is missing. Exclusions/Imputations require researcher approval.")
            
        idx = raw_names_upper.index(ch_upper)
        selected_signals.append(raw_signals[idx, :])
        
    return np.vstack(selected_signals)
