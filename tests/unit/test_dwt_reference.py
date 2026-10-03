import numpy as np
import pywt
import pytest
from seizure_detection.preprocessing.dwt import apply_dwt, process_multichannel_dwt

def reference_dwt(signal: np.ndarray) -> np.ndarray:
    """
    Independent reference implementation of DWT reconstruction for db4 level 5.
    Explicitly builds up the signal using IDWT step-by-step.
    """
    coeffs = pywt.wavedec(signal, 'db4', level=5)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    # Reconstruct level by level, trimming to the exact length of the next detail coeff
    # Level 5 -> 4
    rec4 = pywt.idwt(cA5, cD5, 'db4')[:len(cD4)]
    # Level 4 -> 3
    rec3 = pywt.idwt(rec4, cD4, 'db4')[:len(cD3)]
    # Level 3 -> 2
    rec2 = pywt.idwt(rec3, cD3, 'db4')[:len(cD2)]
    # Level 2 -> 1
    rec1 = pywt.idwt(rec2, np.zeros_like(cD2), 'db4')[:len(cD1)]
    # Level 1 -> 0
    rec0 = pywt.idwt(rec1, np.zeros_like(cD1), 'db4')[:len(signal)]
    
    return rec0

def test_dwt_numerical_equivalence():
    """
    Validates that apply_dwt produces the exact same numerical output as the 
    reference implementation without using the exact same library path (waverec).
    """
    # Create synthetic signal (e.g. 1024 samples)
    np.random.seed(42)
    signal = np.random.randn(1024)
    
    production_out = apply_dwt(signal)
    reference_out = reference_dwt(signal)
    
    assert production_out.shape == (1024,)
    assert np.allclose(production_out, reference_out, atol=1e-10)
    assert np.isfinite(production_out).all()
    assert production_out.dtype == np.float64
