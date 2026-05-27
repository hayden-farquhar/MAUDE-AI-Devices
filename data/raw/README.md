# Raw Data

Raw data files are not included in this repository due to size (~844 MB for MAUDE JSONL files).

## How to obtain

### FDA AI/ML-Enabled Device List

Downloaded automatically by `scripts/01_download_device_list.py`. The Excel file is available at:

https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices

Expected output: `fda_aiml_devices.xlsx` (~127 KB)

### MAUDE Reports

Downloaded automatically by `scripts/02_query_maude.py` via the openFDA Device Event API. No API key is required for rates under 240 requests per minute.

API documentation: https://open.fda.gov/apis/device/event/

Expected output: `maude_by_code/` directory containing one JSONL file per product code (~168 files, ~844 MB total). The script is resumable and will skip product codes already downloaded.

Estimated download time: ~2 hours.
