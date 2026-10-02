import pytest
import tempfile
import os
from seizure_detection.data.summary_parser import parse_chbmit_summary, SummaryParseError

def test_parse_chbmit_summary_valid():
    content = """
Data Sampling Rate: 256 Hz
Channels in EDF Files:
...

File Name: chb01_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
File Start Time: 14:43:12
File End Time: 15:43:12
Number of Seizures in File: 2
Seizure 1 Start Time: 1467 seconds
Seizure 1 End Time: 1494 seconds
Seizure 2 Start Time: 1732 seconds
Seizure 2 End Time: 1772 seconds
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(content)
        f_name = f.name
        
    try:
        annotations = parse_chbmit_summary(f_name)
        assert len(annotations) == 3
        assert annotations["chb01_01.edf"] == []
        assert annotations["chb01_03.edf"] == [(2996, 3036)]
        assert annotations["chb01_04.edf"] == [(1467, 1494), (1732, 1772)]
    finally:
        os.unlink(f_name)

def test_parse_chbmit_summary_mismatch_raises_error():
    content = """
File Name: chb01_03.edf
Number of Seizures in File: 2
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(content)
        f_name = f.name
        
    try:
        with pytest.raises(SummaryParseError, match="Mismatch between number of seizures"):
            parse_chbmit_summary(f_name)
    finally:
        os.unlink(f_name)
