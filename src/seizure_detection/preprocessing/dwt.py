import pywt
import numpy as np

def apply_dwt(signal: np.ndarray, wavelet: str = 'db4', level: int = 5) -> np.ndarray:
    """
    Applies Discrete Wavelet Transform (DWT) to filter the EEG signal.
    Retains only cD5, cD4, cD3, and cA5 components as specified in the paper.
    
    Args:
        signal (np.ndarray): 1D array of EEG signal data.
        wavelet (str): Mother wavelet type.
        level (int): Decomposition level.
        
    Returns:
        np.ndarray: Reconstructed filtered signal.
    """
    if len(signal.shape) != 1:
        raise ValueError("DWT expects a 1D signal array.")
        
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # coeffs order: [cA5, cD5, cD4, cD3, cD2, cD1]
    # We zero out cD2 and cD1 (which correspond to 32-64 Hz and 64-128 Hz)
    # The paper retains cD5 (4-8 Hz), cD4 (8-16 Hz), cD3 (16-32 Hz), cA5 (0-4 Hz)
    
    # Zero out cD1 (index -1) and cD2 (index -2)
    coeffs[-1] = np.zeros_like(coeffs[-1])
    coeffs[-2] = np.zeros_like(coeffs[-2])
    
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    
    # waverec might return a signal with length + 1 due to padding
    return reconstructed_signal[:len(signal)]

def process_multichannel_dwt(signals: np.ndarray, wavelet: str = 'db4', level: int = 5) -> np.ndarray:
    """
    Applies DWT filtering to all channels in a 2D EEG array.
    
    Args:
        signals (np.ndarray): Shape (n_channels, n_samples)
        wavelet (str): Mother wavelet type.
        level (int): Decomposition level.
        
    Returns:
        np.ndarray: Filtered signals of same shape.
    """
    if len(signals.shape) != 2:
        raise ValueError("Expected 2D array of shape (channels, samples).")
        
    filtered = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        filtered[i, :] = apply_dwt(signals[i, :], wavelet, level)
        
    return filtered
