import pytest
from seizure_detection.windowing.annotation_mapper import map_annotations_to_windows

def test_map_annotations_to_windows():
    # 20 seconds total, 4s windows -> 5 windows (0-4, 4-8, 8-12, 12-16, 16-20)
    # Seizure from 5 to 11
    # Window 0-4: NON_SEIZURE
    # Window 4-8: UNRESOLVED_BOUNDARY (crosses start 5)
    # Window 8-12: UNRESOLVED_BOUNDARY (crosses end 11)
    # Window 12-16: NON_SEIZURE
    # Window 16-20: NON_SEIZURE
    
    seizure_intervals = [(5, 11)]
    windows = map_annotations_to_windows(seizure_intervals, recording_duration_sec=20, window_sec=4)
    
    assert len(windows) == 5
    assert windows[0]["status"] == "NON_SEIZURE"
    assert windows[1]["status"] == "UNRESOLVED_BOUNDARY"
    assert windows[2]["status"] == "UNRESOLVED_BOUNDARY"
    assert windows[3]["status"] == "NON_SEIZURE"
    assert windows[4]["status"] == "NON_SEIZURE"

def test_map_annotations_exact_boundaries():
    # 12 seconds total. Seizure exactly 4 to 8.
    # Window 0-4: NON_SEIZURE
    # Window 4-8: SEIZURE
    # Window 8-12: NON_SEIZURE
    
    seizure_intervals = [(4, 8)]
    windows = map_annotations_to_windows(seizure_intervals, recording_duration_sec=12, window_sec=4)
    
    assert len(windows) == 3
    assert windows[0]["status"] == "NON_SEIZURE"
    assert windows[1]["status"] == "SEIZURE"
    assert windows[2]["status"] == "NON_SEIZURE"
