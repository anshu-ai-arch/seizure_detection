from typing import List, Tuple, Dict

def map_annotations_to_windows(
    seizure_intervals: List[Tuple[int, int]], 
    recording_duration_sec: int,
    window_sec: int = 4
) -> List[Dict[str, any]]:
    """
    Identifies the label status of 4-second windows across a continuous recording.
    
    Args:
        seizure_intervals: List of (start_sec, end_sec) for verified seizures.
        recording_duration_sec: Total duration of the recording in seconds.
        window_sec: Duration of each window (default 4s).
        
    Returns:
        List of dictionaries with keys:
            - start_sec: window start
            - end_sec: window end
            - status: "SEIZURE", "NON_SEIZURE", "UNRESOLVED_BOUNDARY"
    """
    windows = []
    for start in range(0, recording_duration_sec, window_sec):
        end = start + window_sec
        if end > recording_duration_sec:
            break  # Drop short final window
            
        is_seizure = False
        is_boundary = False
        
        for s_start, s_end in seizure_intervals:
            # Check overlap
            if start < s_end and end > s_start:
                # If fully inside the seizure annotation
                if start >= s_start and end <= s_end:
                    is_seizure = True
                else:
                    is_boundary = True
                    break
                    
        if is_boundary:
            status = "UNRESOLVED_BOUNDARY"
        elif is_seizure:
            status = "SEIZURE"
        else:
            # Note: Being outside a listed annotation doesn't strictly mean seizure-free
            # without full dataset validation, but we mark it tentatively as non-seizure.
            status = "NON_SEIZURE"
            
        windows.append({
            "start_sec": start,
            "end_sec": end,
            "status": status
        })
        
    return windows
