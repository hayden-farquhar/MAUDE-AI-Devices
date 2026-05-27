# Data Dictionary

## data/processed/fda_aiml_devices.csv

FDA AI/ML-Enabled Device List, cleaned and standardized.

| Variable | Type | Description |
|----------|------|-------------|
| `clearance_date` | date (YYYY-MM-DD) | Date of FDA clearance/approval |
| `submission_number` | string | FDA submission number (e.g., K231207, DEN130013) |
| `device_name` | string | Device trade name |
| `company` | string | Manufacturer name (as listed by FDA) |
| `panel` | string | FDA medical specialty panel (e.g., Radiology, Cardiovascular) |
| `product_code` | string | FDA three-letter product code (e.g., QIH, LNH) |
| `submission_type` | string | Regulatory pathway: K (510(k)), DEN (De Novo), P (PMA) |
| `company_clean` | string | Standardized uppercase company name for matching |

**Source:** https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices

## data/processed/linked_reports.parquet

MAUDE adverse event reports linked to specific AI/ML devices.

| Variable | Type | Description |
|----------|------|-------------|
| `report_number` | string | Unique MAUDE report identifier |
| `event_type` | string | Event type code (e.g., Malfunction, Injury, Death) |
| `date_received` | date | Date report was received by FDA |
| `product_code_queried` | string | Product code used to retrieve this report |
| `device_report_product_code` | string | Product code as reported in MAUDE |
| `brand_name` | string | Device brand name from MAUDE report |
| `generic_name` | string | Device generic name from MAUDE report |
| `manufacturer_d_name` | string | Manufacturer name from MAUDE report |
| `model_number` | string | Device model number |
| `catalog_number` | string | Device catalog number |
| `narrative` | string | Free-text adverse event narrative |
| `date_of_event` | date | Date the adverse event occurred |
| `report_source_code` | string | Source of report (Manufacturer, User facility, Voluntary) |
| `type_of_report` | string | Initial or supplemental report |
| `product_problem_flag` | string | Whether a product problem was reported |
| `event_location` | string | Location where event occurred |
| `matched_submission` | string | FDA submission number of matched AI/ML device |
| `matched_company` | string | Company name of matched device |
| `matched_device` | string | Device name of matched device |
| `matched_panel` | string | Medical specialty panel of matched device |
| `matched_clearance_date` | date | Clearance date of matched device |
| `match_type` | string | Linkage method: "manufacturer" or "manufacturer+brand" |
| `match_score` | float | Fuzzy match score (0-100) for manufacturer name |

**Source:** openFDA Device Event API, linked via scripts 02-03.

## data/processed/classification_progress.csv

LLM classification results for all linked reports (v2 prompt).

| Variable | Type | Description |
|----------|------|-------------|
| `report_number` | string | MAUDE report identifier (links to linked_reports) |
| `categories` | string | Comma-separated failure mode categories (e.g., "B,G") |
| `primary_category` | string | Single primary failure mode category (A-G or U) |
| `confidence` | string | LLM confidence: high, medium, or low |
| `ai_specific` | boolean | True if failure is related to AI/ML functionality |

**Taxonomy codes:** A=Data input, B=Algorithmic/processing, C=Output/interpretation, D=User interaction, E=Infrastructure/integration, F=Hardware, G=Patient harm, U=Uninformative.

## data/processed/classified_reports.parquet

Full classified corpus: linked_reports merged with classification_progress on `report_number`. Contains all columns from both datasets.

## data/processed/linkage_summary.csv

Per-device linkage statistics.

| Variable | Type | Description |
|----------|------|-------------|
| `submission_number` | string | FDA submission number |
| `device_name` | string | Device trade name |
| `company` | string | Manufacturer |
| `panel` | string | Medical specialty |
| `product_code` | string | FDA product code |
| `n_linked` | integer | Number of MAUDE reports linked to this device |
| `match_types` | string | Linkage methods used |

## data/processed/validation_sample_for_review.csv

200-report stratified validation sample with both LLM and manual (physician) labels.

| Variable | Type | Description |
|----------|------|-------------|
| `row_number` | integer | Sequential row number (1-200) |
| `report_number` | string | MAUDE report identifier |
| `device` | string | Matched device name |
| `specialty` | string | Medical specialty panel |
| `narrative` | string | Full MAUDE narrative text |
| `manual_categories` | string | Physician-assigned categories (comma-separated) |
| `manual_primary` | string | Physician-assigned primary category |
| `manual_ai_specific` | boolean | Physician assessment of AI-specificity |
