import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_summary(audit_file: str, manifest_file: str):
    audit_path = Path(audit_file)
    manifest_path = Path(manifest_file)
    
    if not audit_path.exists():
        logger.error(f"Audit file not found: {audit_path}")
        return
        
    audit_df = pd.read_csv(audit_path)
    
    # 1. Total eligible and excluded EDFs
    total_edfs = len(audit_df)
    
    # Let's define eligible as sfreq_valid, channels_valid, and annotations_valid
    # The auditor stores bool strings or bools.
    audit_df["is_eligible"] = audit_df["sfreq_valid"].astype(bool) & audit_df["channels_valid"].astype(bool) & audit_df["annotations_valid"].astype(bool)
    
    eligible_edfs = audit_df["is_eligible"].sum()
    excluded_edfs = total_edfs - eligible_edfs
    
    print("=== DATASET SUMMARY ===")
    print(f"1. Total EDFs processed: {total_edfs}")
    print(f"   Eligible: {eligible_edfs}")
    print(f"   Excluded: {excluded_edfs}")
    
    # 2. Exclusion counts grouped by reason
    if excluded_edfs > 0:
        print("\n2. Exclusion counts by reason:")
        exclusions = audit_df[~audit_df["is_eligible"]]
        reason_counts = exclusions["diagnostic_reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"   - {reason}: {count}")
    
    # 3. Recording counts per patient
    print("\n3. Eligible recording counts per patient:")
    eligible_df = audit_df[audit_df["is_eligible"]]
    patient_counts = eligible_df["patient_id"].value_counts()
    for pid, count in patient_counts.items():
        print(f"   - {pid}: {count}")
        
    if not manifest_path.exists():
        logger.warning(f"Manifest file not found: {manifest_path}. Cannot generate window statistics.")
        return
        
    manifest_df = pd.read_csv(manifest_path)
    
    # 4. Window counts per patient
    print("\n4. Window counts per patient:")
    win_patient_counts = manifest_df["patient_id"].value_counts()
    for pid, count in win_patient_counts.items():
        print(f"   - {pid}: {count}")
        
    # 5. Total windows per status
    print("\n5. Total windows per status:")
    status_counts = manifest_df["status"].value_counts()
    for status, count in status_counts.items():
        print(f"   - {status}: {count}")
        
    # 6. Seizure/non-seizure ratio
    n_seizure = (manifest_df["status"] == "SEIZURE").sum()
    n_non_seizure = (manifest_df["status"] == "NON_SEIZURE").sum()
    if n_non_seizure > 0:
        ratio = n_seizure / n_non_seizure
        print(f"\n6. Seizure/Non-seizure ratio (resolved): {ratio:.4f} ({n_seizure} / {n_non_seizure})")
    else:
        print("\n6. Seizure/Non-seizure ratio: N/A (no non-seizure windows)")
        
    # 7. Unresolved boundary-window count
    n_unresolved = (manifest_df["status"] == "UNRESOLVED_BOUNDARY").sum()
    print(f"\n7. Unresolved boundary-window count: {n_unresolved}")
    
    # 8. Missing/invalid/duplicate manifest entries
    duplicates = manifest_df.duplicated(subset=["window_file"]).sum()
    print(f"\n8. Duplicate manifest window entries: {duplicates}")
    
    # 10. Record of recordings whose annotation status could not be established
    print("\n10. Recordings with unestablished annotations (unlisted or parse failed):")
    unlisted = audit_df[audit_df["is_listed_in_summary"] == False]
    for _, row in unlisted.iterrows():
        print(f"   - {row['patient_id']}/{row['filename']}: {row['diagnostic_reason']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-file", type=str, required=True)
    parser.add_argument("--manifest-file", type=str, required=True)
    args = parser.parse_args()
    generate_summary(args.audit_file, args.manifest_file)
