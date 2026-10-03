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

def inventory_shards(out_dir: str):
    out_path = Path(out_dir)
    shards_dir = out_path / "shards"
    
    if not shards_dir.exists():
        logger.error(f"Shards directory not found: {shards_dir}")
        return
        
    zip_files = list(shards_dir.glob("*.zip"))
    
    total_valid = 0
    total_invalid = 0
    
    for zf in zip_files:
        sha_file = shards_dir / f"{zf.stem}.sha256"
        
        is_zip_valid = verify_zip(zf)
        
        if not is_zip_valid:
            logger.error(f"Invalid ZIP archive: {zf.name}")
            total_invalid += 1
            continue
            
        if sha_file.exists():
            computed = compute_sha256(zf)
            with open(sha_file, "r") as f:
                expected = f.read().split()[0]
                
            if computed != expected:
                logger.error(f"Checksum mismatch for {zf.name}: expected {expected}, got {computed}")
                total_invalid += 1
                continue
        else:
            logger.warning(f"No checksum file found for {zf.name}")
            
        logger.info(f"Verified {zf.name} (Valid)")
        total_valid += 1
        
    logger.info(f"Inventory complete. Valid shards: {total_valid}, Invalid shards: {total_invalid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/processed_data", help="Output directory to inventory")
    args = parser.parse_args()
    inventory_shards(args.out_dir)
