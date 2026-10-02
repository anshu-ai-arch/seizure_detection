import re
from pathlib import Path
from typing import Dict, List, Tuple

class SummaryParseError(Exception):
    pass

def parse_chbmit_summary(file_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses a CHB-MIT summary file to extract seizure start and end times for each EDF.
    
    Args:
        file_path (str): Path to the patient's summary.txt file.
        
    Returns:
        Dict[str, List[Tuple[int, int]]]: A dictionary mapping EDF filenames to a list 
        of seizure intervals (start_sec, end_sec). If an EDF is listed but has 0 seizures,
        it maps to an empty list [].
    """
    if not Path(file_path).exists():
        raise SummaryParseError(f"Summary file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Split content into blocks by file name. 
    # Example: "File Name: chb01_01.edf"
    blocks = re.split(r"File Name:\s+", content)
    
    annotations = {}
    
    for block in blocks[1:]: # Skip the first chunk which is header info
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
            
        filename = lines[0].split()[0]
        seizure_intervals = []
        
        # Look for Number of Seizures
        num_seizures = 0
        for line in lines:
            if line.startswith("Number of Seizures in File:"):
                try:
                    num_seizures = int(line.split(":")[-1].strip())
                except ValueError:
                    raise SummaryParseError(f"Could not parse number of seizures in {filename}")
                break
                
        # Look for Seizure Start/End Times
        start_times = []
        end_times = []
        for line in lines:
            if "Seizure" in line and "Start Time:" in line:
                try:
                    start = int(re.search(r"(\d+)\s+seconds", line).group(1))
                    start_times.append(start)
                except AttributeError:
                    raise SummaryParseError(f"Could not parse start time in {filename}: {line}")
            elif "Seizure" in line and "End Time:" in line:
                try:
                    end = int(re.search(r"(\d+)\s+seconds", line).group(1))
                    end_times.append(end)
                except AttributeError:
                    raise SummaryParseError(f"Could not parse end time in {filename}: {line}")
                    
        if len(start_times) != num_seizures or len(end_times) != num_seizures:
            raise SummaryParseError(f"Mismatch between number of seizures ({num_seizures}) and annotated times in {filename}")
            
        for s, e in zip(start_times, end_times):
            if s >= e:
                raise SummaryParseError(f"Invalid seizure interval {s}-{e} in {filename}")
            seizure_intervals.append((s, e))
            
        annotations[filename] = seizure_intervals
        
    return annotations
