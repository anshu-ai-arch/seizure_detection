import argparse
from pathlib import Path
import json
import zipfile
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_zip(filepath: Path) -> bool:
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            return z.testzip() is None
    except Exception:
        return False

def compute_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def inventory_shards(out_dir: str, data_dir: str = None) -> dict:
    out_path = Path(out_dir)
    shards_dir = out_path / "shards"
    
    report = {
        "valid_shards": [],
        "invalid_shards": [],
        "manifest_coverage_windows": 0,
        "completed_recordings_count": 0,
        "incomplete_or_missing": []
    }

    completed_log = out_path / "completed_recordings.txt"
    completed_recordings = set()
    if completed_log.exists():
        with open(completed_log, "r") as f:
            completed_recordings = set(line.strip() for line in f if line.strip())
    report["completed_recordings_count"] = len(completed_recordings)

    if not shards_dir.exists():
        logger.error(f"Shards directory not found: {shards_dir}")
        report["incomplete_or_missing"] = "unavailable (shards directory not found)"
        report_path = out_path / "recovery_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        return report
        
    zip_files = sorted([z for z in shards_dir.glob("*.zip") if not z.name.endswith("_temp.zip")])
    
    total_manifest_windows = 0
    valid_manifest_recordings = set()

    for zf in zip_files:
        sha_file = shards_dir / f"{zf.stem}.sha256"
        is_zip_valid = verify_zip(zf)
        
        shard_info = {
            "shard": zf.name,
            "verified_zip": is_zip_valid,
            "checksum_match": False,
            "expected_checksum": None,
            "computed_checksum": None,
            "manifest_windows": 0,
            "recordings": []
        }
        
        if not is_zip_valid:
            logger.error(f"Invalid ZIP archive: {zf.name}")
            report["invalid_shards"].append(shard_info)
            continue
            
        if not sha_file.exists():
            logger.warning(f"No checksum file found for {zf.name}")
            report["invalid_shards"].append(shard_info)
            continue

        computed = compute_sha256(zf)
        sha_tokens = sha_file.read_text().strip().split()
        if not sha_tokens:
            logger.error(f"Empty checksum file for {zf.name}")
            report["invalid_shards"].append(shard_info)
            continue

        expected = sha_tokens[0].lower()
        shard_info["expected_checksum"] = expected
        shard_info["computed_checksum"] = computed
            
        if computed.lower() != expected:
            logger.error(f"Checksum mismatch for {zf.name}: expected {expected}, got {computed}")
            report["invalid_shards"].append(shard_info)
            continue
        
        shard_info["checksum_match"] = True

        # Inspect manifest rows inside the verified zip
        shard_windows = 0
        shard_recordings = set()
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                manifest_files = [n for n in z.namelist() if n.endswith("_manifest.csv") or n == "manifest.csv"]
                for m_file in manifest_files:
                    with z.open(m_file) as mf:
                        import io
                        import csv
                        reader = csv.DictReader(io.TextIOWrapper(mf, encoding='utf-8'))
                        for row in reader:
                            shard_windows += 1
                            pid = row.get("patient_id")
                            fname = row.get("filename")
                            if pid and fname:
                                shard_recordings.add(f"{pid}_{fname}")

            shard_info["manifest_windows"] = shard_windows
            shard_info["recordings"] = sorted(list(shard_recordings))
            total_manifest_windows += shard_windows
            valid_manifest_recordings.update(shard_recordings)
            report["valid_shards"].append(shard_info)
        except Exception as e:
            logger.error(f"Error reading manifest from {zf.name}: {e}")
            shard_info["error"] = str(e)
            report["invalid_shards"].append(shard_info)

    report["manifest_coverage_windows"] = total_manifest_windows

    # Determine incomplete or missing recordings
    unsharded_completed = sorted(list(completed_recordings - valid_manifest_recordings))

    if data_dir and Path(data_dir).exists():
        dataset_path = Path(data_dir)
        all_dataset_recordings = {
            f"{p.name}_{edf.name}"
            for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("chb")
            for edf in p.glob("*.edf")
        }
        report["incomplete_or_missing"] = sorted(list(all_dataset_recordings - valid_manifest_recordings))
    elif unsharded_completed:
        report["incomplete_or_missing"] = unsharded_completed
    else:
        report["incomplete_or_missing"] = "unavailable (data_dir not provided)"

    report_path = out_path / "recovery_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Inventory complete. Valid shards: {len(report['valid_shards'])}, Invalid shards: {len(report['invalid_shards'])}")
    logger.info(f"Machine-readable report saved to {report_path}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory and verify processed shards")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/processed_data", help="Output directory to inventory")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional dataset directory to check completeness")
    args = parser.parse_args()
    inventory_shards(out_dir=args.out_dir, data_dir=args.data_dir)
