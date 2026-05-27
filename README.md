# The Surveillance Gap: Failure Mode Taxonomy for FDA-Cleared AI/ML Medical Devices in MAUDE

Code and data repository for: **The Surveillance Gap: Failure Mode Taxonomy and Post-Market Monitoring Adequacy of FDA-Cleared AI/ML Medical Devices in MAUDE**

Hayden Farquhar MBBS MPHTM, Independent researcher, Finley, NSW, Australia. ORCID: [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)

Preprint: to be posted

## Overview

This repository contains the analysis code, processed datasets, and failure mode taxonomy for a study linking all 1,430 devices on the FDA's AI/ML-Enabled Device List to 21,902 MAUDE adverse event reports (2020-2026). Reports were classified into an eight-category AI-specific failure mode taxonomy using GPT-4o-mini, validated against 200 manually labeled reports (Cohen's kappa 0.61, macro F1 0.70). The study finds that 91.3% of FDA-listed AI/ML devices have zero MAUDE reports, and only 24.0% of reports on AI-enabled devices describe failures specific to the AI/ML functionality.

## Data Sources

| Source | URL | Access | License |
|--------|-----|--------|---------|
| FDA AI/ML-Enabled Device List | https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices | Free download | Public domain (US government work) |
| FDA MAUDE (via openFDA) | https://open.fda.gov/apis/device/event/ | Free REST API (<240 req/min without key) | Public domain (US government work) |

Raw MAUDE JSONL files (~844 MB) are not included in this repository due to size. They can be reproduced by running scripts 01 and 02, or downloaded directly from the openFDA API.

## Requirements

- Python 3.10+
- ~10 GB disk space (for raw MAUDE downloads)
- OpenAI API key (for script 04 only; set as `OPENAI_API_KEY` environment variable)

```bash
pip install -r requirements.txt
```

## Reproduction

Scripts are numbered in execution order. Scripts 01-03 acquire and link data; script 04 requires an OpenAI API key (~$7 per run); scripts 05-08 are analysis and visualization.

```bash
# Step 1: Download FDA AI/ML device list
python scripts/01_download_device_list.py

# Step 2: Query openFDA MAUDE for all AI device product codes
# Note: Takes ~2 hours. Resumable — skips codes already downloaded.
python scripts/02_query_maude.py

# Step 3: Link devices to MAUDE reports
python scripts/03_link_devices_reports.py

# Step 3b (optional): Verify linkage on 50-device sample
python scripts/03b_verify_linkage.py

# Step 4: Classify reports using GPT-4o-mini
# Requires OPENAI_API_KEY. ~25 min, ~$7. Resumable via checkpointing.
python scripts/04_classify_failures.py

# Step 5: Create validation sample (200 reports, stratified)
python scripts/05_validate_classification.py

# Step 5b: Compute validation metrics (after manual labeling)
python scripts/05_validate_classification.py metrics

# Step 6: Surveillance gap analysis
python scripts/06_surveillance_gap.py

# Step 7: AI-specific failure deep dive
python scripts/07_ai_failure_deep_dive.py

# Step 8: Generate manuscript figures and tables
python scripts/08_manuscript_figures.py
```

**Estimated total runtime:** ~3 hours (dominated by MAUDE API queries in step 2 and LLM classification in step 4). Steps 5-8 complete in under 2 minutes total.

**To reproduce from processed data only** (skipping data acquisition): The `data/processed/` directory contains all intermediate outputs needed to run scripts 05-08 directly.

## Script Descriptions

| Script | Description | Inputs | Outputs |
|--------|-------------|--------|---------|
| `01_download_device_list.py` | Downloads FDA AI/ML-Enabled Device List Excel file; parses and cleans to CSV/Parquet | FDA website | `data/raw/fda_aiml_devices.xlsx`, `data/processed/fda_aiml_devices.csv` |
| `02_query_maude.py` | Queries openFDA MAUDE API for all 168 product codes with date-partitioned pagination | `data/processed/fda_aiml_devices.csv` | `data/raw/maude_by_code/*.jsonl`, `data/processed/maude_query_summary.csv` |
| `03_link_devices_reports.py` | Multi-stage linkage: product code + manufacturer name fuzzy matching (threshold >= 85) | Device list + MAUDE JSONL files | `data/processed/linked_reports.parquet`, `data/processed/linkage_summary.csv` |
| `03b_verify_linkage.py` | Verifies linkage precision on 50-device random sample | `data/processed/linked_reports.parquet` | `data/processed/verification_sample.csv` |
| `04_classify_failures.py` | GPT-4o-mini few-shot multi-label classification into 8-category taxonomy | `data/processed/linked_reports.parquet` | `data/processed/classification_progress.csv`, `data/processed/classified_reports.parquet` |
| `05_validate_classification.py` | Creates stratified 200-report validation sample; computes Cohen's kappa and per-category F1 | `data/processed/classification_progress.csv` | `data/processed/validation_sample.csv`, `outputs/tables/validation_metrics.csv` |
| `06_surveillance_gap.py` | Surveillance adequacy analysis: zero-report rates, temporal trends, specialty/manufacturer stratification | Device list + linked/classified reports | `outputs/tables/surveillance_summary.csv`, `outputs/tables/device_report_counts.csv`, and 4 more tables |
| `07_ai_failure_deep_dive.py` | AI-specific failure analysis: B/C subset, patient harm rates, narrative quality, exemplar extraction | Classified reports | `outputs/tables/ai_failure_exemplars.csv`, `outputs/tables/ai_vs_generic_comparison.csv`, and 2 more tables |
| `08_manuscript_figures.py` | Generates 5 publication figures (PNG + PDF) and Table 1 | All processed data | `outputs/figures/fig1-5.*`, `outputs/tables/table1_cohort.csv` |

## Failure Mode Taxonomy

Eight-category AI-specific failure mode taxonomy with multi-label assignment:

| Code | Category | Description | AI-relevant? |
|------|----------|-------------|:---:|
| A | Data input | Poor image/signal quality, sensor malfunction, wrong positioning | Yes |
| B | Algorithmic/processing | False positive/negative, misclassification, software bug, firmware failure | Yes |
| C | Output/interpretation | Misleading display, wrong patient metadata on results, report generation error | Yes |
| D | User interaction | Automation bias, alert fatigue, UI confusion, training gap | Interaction |
| E | Infrastructure/integration | Network loss, EHR/PACS error, DICOM issue, interoperability failure | No |
| F | Hardware | Display failure, power issue, mechanical breakage, overheating | No |
| G | Patient harm | Death, injury, delayed diagnosis (always co-assigned with causal category) | — |
| U | Uninformative | Pure regulatory boilerplate, no event description | — |

An additional binary flag (`ai_specific`) distinguishes failures related to the AI/ML component from generic device failures on AI-enabled platforms.

## Outputs

### Figures

| File | Paper reference |
|------|----------------|
| `outputs/figures/fig1_report_distribution.png` | Figure 1: Surveillance gap (91.3% zero-report rate) |
| `outputs/figures/fig2_failure_modes.png` | Figure 2: Failure mode distribution with AI-specific overlay |
| `outputs/figures/fig3_temporal_trends.png` | Figure 3: Device clearances and MAUDE reports 2015-2025 |
| `outputs/figures/fig4_specialty_heatmap.png` | Figure 4: Failure modes by medical specialty |
| `outputs/figures/fig5_surveillance_by_year.png` | Figure 5: Surveillance rate by clearance year cohort |

### Tables

| File | Paper reference |
|------|----------------|
| `outputs/tables/table1_cohort.csv` | Table 1: Study cohort characteristics |
| `outputs/tables/surveillance_summary.csv` | Device-level surveillance metrics |
| `outputs/tables/specialty_surveillance.csv` | Table 2: Specialty-stratified surveillance |
| `outputs/tables/validation_metrics.csv` | Table 3: Classifier validation metrics |
| `outputs/tables/top20_devices.csv` | Supplementary Table 1: Top 20 devices |
| `outputs/tables/ai_failure_exemplars.csv` | Supplementary Table 2: Exemplar narratives |
| `outputs/tables/ai_vs_generic_comparison.csv` | AI-specific vs generic failure comparison |

## Data Dictionary

See [`data_dictionary.md`](data_dictionary.md) for variable definitions across all datasets.

## Citation

If you use this code or data, please cite:

```
Farquhar H. The Surveillance Gap: Failure Mode Taxonomy and Post-Market Monitoring
Adequacy of FDA-Cleared AI/ML Medical Devices in MAUDE. 2026.
```

## License

Code: MIT License. Data and documentation: CC-BY 4.0.

See [LICENSE](LICENSE) for full text.
