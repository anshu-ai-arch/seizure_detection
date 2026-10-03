from typing import List, Tuple, Dict

def map_annotations_to_windows(
    seizure_intervals: List[Tuple[int, int]],
    recording_duration_sec: float,
    window_sec: int = 4,
    overlap_seizure: float = 0.5,
    overlap_non_seizure: float = 0.0
) -> List[Dict[str, any]]:
    """
    Identifies the label status of 4-second windows across a continuous recording,
    applying paper-specified overlaps (50% for seizure, 0% for non-seizure).
    """
    windows = []
    start = 0.0

    while start + window_sec <= recording_duration_sec:
        end = start + window_sec
        is_seizure = False
        is_boundary = False

        for s_start, s_end in seizure_intervals:
            if start < s_end and end > s_start:
                if start >= s_start and end <= s_end:
                    is_seizure = True
                else:
                    is_boundary = True
                break

        if is_boundary:
            status = "UNRESOLVED_BOUNDARY"
            stride = window_sec * (1 - overlap_non_seizure)
        elif is_seizure:
            status = "SEIZURE"
            stride = window_sec * (1 - overlap_seizure)
        else:
            status = "NON_SEIZURE"
            stride = window_sec * (1 - overlap_non_seizure)

        windows.append({
            "start_sec": start,
            "end_sec": end,
            "status": status
        })

        start += stride

    return windows
