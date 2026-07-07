
# Vancomycin–AKI Evidence Extraction Pipeline (MIMIC-IV)

## Overview

This repository contains a structured **evidence extraction pipeline** for identifying causality-related signals associated with Vancomycin-induced Acute Kidney Injury (AKI) using MIMIC-IV data.

The primary objective at this stage is to systematically extract and organise relevant clinical evidence that may later support pharmacovigilance analysis.

---

## Aim

To develop a **transparent, reproducible, and evidence-grounded extraction pipeline** that captures key clinical features required for downstream causality assessment.

This aligns with a staged approach:

1. **Stage 1 (Current)**: Evidence extraction
2. Stage 2: Evidence interpretation and reasoning
3. Stage 3: Causality scoring (e.g., Naranjo, LLM-based methods)

---

## Data Sources

The pipeline uses the following MIMIC-IV tables:

* `admissions.csv`
* `prescriptions.csv`
* `labevents.csv`
* `diagnoses_icd.csv`
* `d_icd_diagnoses.csv`
* Clinical notes (discharge summaries)

Licencing and authorisation is required for these datasets
---

## Pipeline Structure

### Phase 1 — Structured Clinical Evidence Extraction

Extracts key structured variables per admission:

* Drug exposure (Vancomycin start/stop time)
* Creatinine measurements (baseline and post-exposure values)
* Time-aligned laboratory trends

### Phase 2 — Alternative Cause Identification (Rule-Based)

Identifies potential confounders using ICD-coded diagnoses:

* Sepsis
* Hypotension
* Dehydration
* Renal disease
* Shock, heart failure, etc.

Also detects co-administered nephrotoxic drugs.

### Phase 3 — Lightweight NLP on Clinical Notes

Extracts relevant sentences from discharge notes using:

* Keyword matching (e.g., “AKI”, “creatinine”, “vancomycin”)
* Simple rule-based classification:

  * Negation
  * Positive evidence phrases
  * Requires manual review

Note: This phase performs **evidence extraction only**.

---

## Output

The pipeline produces structured outputs in:

### 1. JSON (Detailed)

* One record per admission
* Includes:

  * Drug exposure timing
  * Creatinine values
  * Extracted note evidence
  * Alternative causes
  * Missing/uncertain fields

### 2. Tabular Summary (CSV/Excel)

* Flattened version of key variables
* Suitable for analysis and modelling

---

## Example Output Fields

* `subject_id`
* `hadm_id`
* `drug_start_time`
* `drug_stop_time`
* `baseline_creatinine`
* `peak_creatinine`
* `temporal_relationship` *(to be refined as pure extraction)*
* `dechallenge`
* `possible_alternative_causes`
* `other_nephrotoxic_drugs`
* `relevant_note_evidence`
* `missing_evidence`

---

## Dechallenge Definition (Current Implementation)

Dechallenge is operationalised as follows:

* Identify peak creatinine after drug exposure
* Check for follow-up creatinine within 72 hours after drug discontinuation
* If creatinine decreases by ≥ 0.3 mg/dL → labelled as **positive dechallenge**
* If no decrease → **not supported**
* If no follow-up data → **unknown**

---

## Important Note on Scope

At this stage, the pipeline:

* ✅ Extracts structured and unstructured clinical evidence
* ❌ Does NOT use LLM reasoning for decision-making

The focus is on building a **reliable evidence layer** before introducing modelling or inference.

---

## Future Work

* Refine outputs to strictly separate **raw evidence vs derived signals**
* Introduce Retrieval-Augmented Generation (RAG) for contextual reasoning
* Implement explainable causality assessment (e.g., Naranjo + LLM hybrid)
* Scale pipeline to full MIMIC dataset

---

## Author

Chinenyengozi Okoronkwo
PhD Researcher — NLP & AI for Pharmacovigilance

---

## Notes

This project is part of ongoing doctoral research focused on:
**Automating pharmacovigilance compliance and causality assessment using NLP and AI.**
