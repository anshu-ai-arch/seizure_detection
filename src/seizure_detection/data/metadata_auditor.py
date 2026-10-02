import pandas as pd
from typing import List, Dict, Any

class MetadataAuditor:
    """
    Records and manages patient and recording eligibility metadata.
    Does not silently drop data; explicitly flags excluded/unresolved statuses.
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        
    def add_recording(
        self,
        patient_id: str,
        filename: str,
        is_listed_in_summary: bool,
        is_patient_excluded: bool,
        sfreq_valid: bool,
        channels_valid: bool,
        duration_sec: float,
        annotations_valid: bool,
        diagnostic_reason: str = ""
    ):
        status = "ELIGIBLE"
        if is_patient_excluded:
            status = "EXCLUDED_BY_PAPER"
        elif not is_listed_in_summary:
            status = "UNRESOLVED_UNLISTED"
        elif not sfreq_valid:
            status = "EXCLUDED_SFREQ"
        elif not channels_valid:
            status = "EXCLUDED_MISSING_CHANNELS"
        elif not annotations_valid:
            status = "EXCLUDED_ANNOTATION_ERROR"
            
        self.records.append({
            "patient_id": patient_id,
            "filename": filename,
            "is_listed": is_listed_in_summary,
            "sfreq_valid": sfreq_valid,
            "channels_valid": channels_valid,
            "annotations_valid": annotations_valid,
            "duration_sec": duration_sec,
            "status": status,
            "reason": diagnostic_reason
        })
        
    def export_csv(self, output_path: str):
        df = pd.DataFrame(self.records)
        df.to_csv(output_path, index=False)
