# ============================================================
# VANCOMYCIN → AKI PIPELINE
# Phase 1: Structured extraction (drug timing, creatinine, KDIGO)
# Phase 2: Rule-based alternative causes (FIXED — uses ICD codes)
# Phase 3: Lightweight NLP on clinical notes
# ============================================================

import pandas as pd
import os
import re
import json
from datetime import timedelta

FOLDER = r"C:\Users\HP\OneDrive\PHD\Mimic Dataset"


# ============================================================
# LOAD FILES
# ============================================================

print("Loading admissions...")
admissions = pd.read_csv(
    os.path.join(FOLDER, "admissions.csv MIMIC IV.gz"),
    compression="gzip",
    low_memory=False
)
print(f"  {len(admissions):,} admissions")


print("Loading prescriptions in chunks...")

chunks = []
chunk_size = 500_000  # adjust based on your RAM

for chunk in pd.read_csv(
    os.path.join(FOLDER, "prescriptions.csv MIMIC IV.gz"),
    compression="gzip",
    chunksize=chunk_size,
    low_memory=False
):
    # Filter early → only keep Vancomycin rows
    vanco_chunk = chunk[
        chunk["drug"].str.contains("vancomycin", case=False, na=False)
    ]
    
    chunks.append(vanco_chunk)

prescriptions = pd.concat(chunks, ignore_index=True)

print(f"  {len(prescriptions):,} filtered prescriptions (Vancomycin only)")

print("Loading lab item codes...")
d_labitems = pd.read_csv(
    os.path.join(FOLDER, "d_labitems.csv.gz"),
    compression="gzip"
)
print(f"  {len(d_labitems):,} lab items")

print("Loading diagnosis codes (for Phase 2)...")
diagnoses_icd = pd.read_csv(
    os.path.join(FOLDER, "diagnoses_icd.csv.gz"),
    compression="gzip",
    low_memory=False
)
print(f"  {len(diagnoses_icd):,} diagnosis records")

d_icd_diagnoses = pd.read_csv(
    os.path.join(FOLDER, "d_icd_diagnoses.csv.gz"),
    compression="gzip"
)
print(f"  {len(d_icd_diagnoses):,} ICD code descriptions")

# Merge diagnosis codes with their readable names — one row per
# diagnosis per admission, now with a human-readable "long_title"
diagnoses_named = diagnoses_icd.merge(
    d_icd_diagnoses,
    on=["icd_code", "icd_version"],
    how="left"
)
print("   Diagnosis codes merged with readable names")


# ── This Ffinds creatinine item codes before loading the huge lab file ──
print("\nFinding creatinine item codes...")
creatinine_items = d_labitems[
    d_labitems["label"].str.contains("creatinine", case=False, na=False)
]
print(creatinine_items[["itemid", "label", "fluid"]].to_string())

creatinine_ids = creatinine_items[
    creatinine_items["fluid"].str.contains("blood", case=False, na=False)
]["itemid"].tolist()

print(f"\nBlood creatinine item IDs: {creatinine_ids}")


# ============================================================
# PHASE 1 — STEP 1: FIND VANCOMYCIN ADMISSIONS
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — STEP 1: Vancomycin Admissions")
print("=" * 55)

vancomycin = prescriptions[
    prescriptions["drug"].str.contains("vancomycin", case=False, na=False)
].copy()

vancomycin["starttime"] = pd.to_datetime(vancomycin["starttime"], errors="coerce")
vancomycin["stoptime"]  = pd.to_datetime(vancomycin["stoptime"],  errors="coerce")

print(f"Vancomycin prescription rows : {len(vancomycin):,}")
print(f"Unique admissions            : {vancomycin['hadm_id'].nunique():,}")
print(f"Unique patients              : {vancomycin['subject_id'].nunique():,}")


# ============================================================
# PHASE 1 — STEP 2: DRUG START AND STOP TIMES
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — STEP 2: Drug Start and Stop Times")
print("=" * 55)

vanco_timing = vancomycin.groupby(["subject_id", "hadm_id"]).agg(
    drug_start_time=("starttime", "min"),
    drug_stop_time=("stoptime", "max"),
    total_doses=("starttime", "count")
).reset_index()

print(f"Admissions with start time   : {vanco_timing['drug_start_time'].notna().sum()}")
print(f"Admissions missing start time: {vanco_timing['drug_start_time'].isna().sum()}")
print(f"Admissions with stop time    : {vanco_timing['drug_stop_time'].notna().sum()}")
print(f"Admissions missing stop time : {vanco_timing['drug_stop_time'].isna().sum()}")

# Take sample of 20 — NOTE: this is a convenience sample (first 20 rows tht contins vancomycin),
# not random or stratified. 
sample_timing = vanco_timing.head(20).copy()
sample_hadm_ids = sample_timing["hadm_id"].tolist()

print(f"\nWorking with 20 admissions:")
print(sample_timing[[
    "subject_id", "hadm_id", "drug_start_time", "drug_stop_time", "total_doses"
]].to_string(index=False))


# ============================================================
# PHASE 1 — STEP 3: LOAD CREATININE VALUES (CHUNKED)
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — STEP 3: Loading Creatinine Lab Values")
print("IMPORTANT: Filtering large file")
print("=" * 55)

CHUNK_SIZE = 500_000
creatinine_rows = []
chunk_count = 0

for chunk in pd.read_csv(
    os.path.join(FOLDER, "labevents.csv.gz"),
    compression="gzip",
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk_count += 1

    filtered = chunk[
        (chunk["itemid"].isin(creatinine_ids)) &
        (chunk["hadm_id"].isin(sample_hadm_ids)) &
        (chunk["valuenum"].notna()) &
        (chunk["valuenum"] > 0) &
        (chunk["valuenum"] < 30)
    ]

    if len(filtered) > 0:
        creatinine_rows.append(filtered)

    if chunk_count % 10 == 0:
        print(f"  Processed {chunk_count * CHUNK_SIZE:,} rows...")

creatinine = pd.concat(creatinine_rows, ignore_index=True)
creatinine["charttime"] = pd.to_datetime(creatinine["charttime"], errors="coerce")
creatinine = creatinine.sort_values(["hadm_id", "charttime"])

print(f"\n✓ Creatinine rows found: {len(creatinine):,}")
print(f"  Admissions with creatinine data: {creatinine['hadm_id'].nunique()}")


# ============================================================
# PHASE 1 — STEP 4: BASELINE AND PEAK CREATININE
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — STEP 4: Baseline and Peak Creatinine")
print("=" * 55)

def get_creatinine_features(hadm_id, drug_start, creatinine_df):
    """
    Baseline = median creatinine in 48hrs BEFORE drug start.
    Peak     = highest creatinine value AFTER drug start.
    """
    patient_creat = creatinine_df[creatinine_df["hadm_id"] == hadm_id].copy()

    if patient_creat.empty:
        return {
            "baseline_creatinine": None,
            "peak_creatinine": None,
            "peak_creatinine_time": None,
            "creatinine_missing": True,
            "creatinine_missing_reason": "No creatinine measurements found"
        }

    if pd.isna(drug_start):
        return {
            "baseline_creatinine": None,
            "peak_creatinine": None,
            "peak_creatinine_time": None,
            "creatinine_missing": True,
            "creatinine_missing_reason": "Drug start time missing"
        }

    baseline_window = patient_creat[
        (patient_creat["charttime"] >= drug_start - timedelta(hours=48)) &
        (patient_creat["charttime"] < drug_start)
    ]
    baseline = (
        round(float(baseline_window["valuenum"].median()), 2)
        if not baseline_window.empty else None
    )

    post_drug = patient_creat[patient_creat["charttime"] >= drug_start]

    if post_drug.empty:
        return {
            "baseline_creatinine": baseline,
            "peak_creatinine": None,
            "peak_creatinine_time": None,
            "creatinine_missing": True,
            "creatinine_missing_reason": "No creatinine after drug start"
        }

    peak_row = post_drug.loc[post_drug["valuenum"].idxmax()]
    peak = round(float(peak_row["valuenum"]), 2)
    peak_time = peak_row["charttime"]

    return {
        "baseline_creatinine": baseline,
        "peak_creatinine": peak,
        "peak_creatinine_time": str(peak_time),
        "creatinine_missing": False,
        "creatinine_missing_reason": None
    }


# ============================================================
# PHASE 1 — STEP 5: AKI SIGNAL (KDIGO CRITERIA)
# ============================================================

def check_aki_signal(baseline, peak):
    """
    KDIGO criteria — AKI = rise of >= 0.3 mg/dL OR >= 1.5x baseline.
    """
    if baseline is None or peak is None:
        return "unknown — missing creatinine data"

    absolute_rise = peak - baseline
    relative_rise = peak / baseline if baseline > 0 else 0

    if absolute_rise >= 0.3 or relative_rise >= 1.5:
        return f"YES — rise of {absolute_rise:.2f} mg/dL ({relative_rise:.1f}x baseline)"
    else:
        return f"NO — rise of only {absolute_rise:.2f} mg/dL ({relative_rise:.1f}x baseline)"


# ============================================================
# PHASE 1 — STEP 6: BUILD PATIENT TIMELINE
# ============================================================

def build_timeline(hadm_id, drug_start, drug_stop, baseline, peak, peak_time, creatinine_df):
    """
    drug start → baseline → peak → drug stop → follow-up creatinine
    """
    timeline = []

    if pd.notna(drug_start):
        timeline.append({"event": "Vancomycin started", "time": str(drug_start), "value": None})

    if baseline is not None:
        timeline.append({"event": "Baseline creatinine", "time": "before drug start", "value": f"{baseline} mg/dL"})

    if peak is not None:
        timeline.append({"event": "Peak creatinine", "time": str(peak_time), "value": f"{peak} mg/dL"})

    if pd.notna(drug_stop):
        timeline.append({"event": "Vancomycin stopped", "time": str(drug_stop), "value": None})

        patient_creat = creatinine_df[creatinine_df["hadm_id"] == hadm_id]
        follow_up = patient_creat[
            (patient_creat["charttime"] > drug_stop) &
            (patient_creat["charttime"] <= drug_stop + timedelta(hours=72))
        ]

        if not follow_up.empty:
            last_creat = round(float(follow_up.iloc[-1]["valuenum"]), 2)
            timeline.append({
                "event": "Follow-up creatinine (within 72hrs)",
                "time": str(follow_up.iloc[-1]["charttime"]),
                "value": f"{last_creat} mg/dL"
            })
        else:
            timeline.append({"event": "Follow-up creatinine", "time": "unavailable", "value": "unknown"})

    return timeline


# ============================================================
# PHASE 1 — RUN FOR ALL 20 ADMISSIONS
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — Building Evidence Table")
print("=" * 55)

results = []

for _, row in sample_timing.iterrows():
    hadm_id    = row["hadm_id"]
    subject_id = row["subject_id"]
    drug_start = row["drug_start_time"]
    drug_stop  = row["drug_stop_time"]

    print(f"  Processing admission {hadm_id}...")

    creat = get_creatinine_features(hadm_id, drug_start, creatinine)
    baseline  = creat["baseline_creatinine"]
    peak      = creat["peak_creatinine"]
    peak_time = creat["peak_creatinine_time"]

    aki = check_aki_signal(baseline, peak)

    timeline = build_timeline(hadm_id, drug_start, drug_stop, baseline, peak, peak_time, creatinine)

    # ── Temporal relationship (only "supported" when AKI is YES) ──
    if "YES" in aki and peak_time:
        peak_dt = pd.to_datetime(peak_time)
        hrs_to_peak = (peak_dt - drug_start).total_seconds() / 3600
        temporal = f"supported — creatinine rose, peak {hrs_to_peak:.0f} hours after drug start"
    elif "NO" in aki:
        temporal = "not supported — no significant creatinine rise"
    elif baseline is None or peak is None:
        temporal = "unclear — missing creatinine data"
    else:
        temporal = "unclear"

    # ── Dechallenge ──
    dechallenge = "unknown"
    if pd.notna(drug_stop) and peak is not None:
        follow_up_data = creatinine[
            (creatinine["hadm_id"] == hadm_id) &
            (creatinine["charttime"] > drug_stop) &
            (creatinine["charttime"] <= drug_stop + timedelta(hours=72))
        ]
        if not follow_up_data.empty:
            follow_up_val = float(follow_up_data.iloc[-1]["valuenum"])
            improvement = peak - follow_up_val
            dechallenge = (
                f"positive — creatinine fell {improvement:.2f} mg/dL after drug stopped"
                if improvement >= 0.3
                else f"not supported — creatinine did not improve (change: {improvement:.2f} mg/dL)"
            )

    # ── Missing evidence ──
    missing = []
    if pd.isna(drug_start):
        missing.append("drug start time")
    if pd.isna(drug_stop):
        missing.append("drug stop time")
    if baseline is None:
        missing.append("baseline creatinine")
    if peak is None:
        missing.append("post-drug creatinine")
    if creat["creatinine_missing"]:
        missing.append(creat["creatinine_missing_reason"])

    results.append({
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "suspected_drug": "Vancomycin",
        "drug_start_time": str(drug_start),
        "drug_stop_time": str(drug_stop),
        "baseline_creatinine": baseline,
        "peak_creatinine": peak,
        "aki_signal": aki,
        "temporal_relationship": temporal,
        "dechallenge": dechallenge,
        "timeline": timeline,
        "missing_evidence": "; ".join(missing) if missing else "none",
    })

print(f"\n Phase 1 complete — processed {len(results)} admissions")

assert len(results) == 20, "Something is wrong — not all 20 admissions processed!"


# ============================================================
# PHASE 1 — SUMMARY CHECK
# ============================================================

temporal_supported     = sum(1 for r in results if r["temporal_relationship"].startswith("supported"))
temporal_not_supported = sum(1 for r in results if r["temporal_relationship"].startswith("not supported"))
temporal_unclear        = sum(1 for r in results if r["temporal_relationship"].startswith("unclear"))

print(f"\nPhase 1 — Temporal relationship breakdown:")
print(f"  Supported     : {temporal_supported}")
print(f"  Not supported : {temporal_not_supported}")
print(f"  Unclear       : {temporal_unclear}")

# Consistency check — every "supported" case must also be AKI = YES
mismatch = [
    r for r in results
    if r["temporal_relationship"].startswith("supported") and "YES" not in r["aki_signal"]
]
print(f"  Consistency check — mismatched cases: {len(mismatch)} ")


# ============================================================
# PHASE 2 — RULE-BASED ALTERNATIVE CAUSE DETECTION 
# Uses diagnoses_icd.csv.gz + d_icd_diagnoses.csv.gz
# (admissions.csv in MIMIC-IV has NO free-text diagnosis column)
# ============================================================

print("\n" + "=" * 55)
print("PHASE 2 — Rule-Based Alternative Cause Detection")
print("=" * 55)

ALTERNATIVE_CAUSE_KEYWORDS = {
    "sepsis":        ["sepsis", "septic"],
    "hypotension":   ["hypotension", "hypotensive"],
    "dehydration":   ["dehydration", "dehydrated", "hypovolemia", "hypovolemic"],
    "contrast":      ["contrast"],
    "renal failure": ["renal failure", "chronic kidney disease", "ckd"],
    "shock":         ["shock"],
    "heart failure": ["heart failure", "chf", "cardiac failure"],
}

NEPHROTOXIC_DRUGS = [
    "gentamicin", "tobramycin", "amikacin",   # aminoglycosides
    "ibuprofen", "naproxen",                   # NSAIDs
    "amphotericin",                             # antifungal
    "furosemide",                                # diuretic
    "lisinopril", "enalapril",                  # ACE inhibitors
    "contrast",                                  # imaging dye
]


def check_alternative_causes(hadm_id, diagnoses_named_df):
    """
    Look at this admission's ICD-coded diagnoses.
    Return alternative causes found, using readable diagnosis names.
    """
    patient_diagnoses = diagnoses_named_df[diagnoses_named_df["hadm_id"] == hadm_id]

    if patient_diagnoses.empty:
        return {"alternative_causes_found": [], "diagnosis_text_available": False}

    all_diagnosis_text = " ".join(
        patient_diagnoses["long_title"].dropna().str.lower().tolist()
    )

    if all_diagnosis_text.strip() == "":
        return {"alternative_causes_found": [], "diagnosis_text_available": False}

    found_causes = []
    for cause_name, keywords in ALTERNATIVE_CAUSE_KEYWORDS.items():
        if any(kw in all_diagnosis_text for kw in keywords):
            found_causes.append(cause_name)

    return {
        "alternative_causes_found": found_causes,
        "diagnosis_text_available": True,
        "diagnosis_text": all_diagnosis_text
    }


def check_other_nephrotoxic_drugs(hadm_id, prescriptions_df):
    """
    Did this patient also receive another drug from our short
    named list of nephrotoxic drugs? Vancomycin excluded.
    """
    patient_drugs = (
        prescriptions_df[prescriptions_df["hadm_id"] == hadm_id]["drug"]
        .dropna().str.lower().unique().tolist()
    )

    found = []
    for drug_text in patient_drugs:
        if "vancomycin" in drug_text:
            continue
        for nephrotoxic in NEPHROTOXIC_DRUGS:
            if nephrotoxic in drug_text and nephrotoxic not in found:
                found.append(nephrotoxic)

    return found


print("\nChecking alternative causes and co-drugs for 20 admissions...\n")

phase2_results = {}

for _, row in sample_timing.iterrows():
    hadm_id = row["hadm_id"]
    print(f"  Checking admission {hadm_id}...")

    alt_causes  = check_alternative_causes(hadm_id, diagnoses_named)
    other_drugs = check_other_nephrotoxic_drugs(hadm_id, prescriptions)

    phase2_results[hadm_id] = {
        "alternative_causes_found": alt_causes["alternative_causes_found"],
        "diagnosis_text_available": alt_causes["diagnosis_text_available"],
        "other_nephrotoxic_drugs": other_drugs,
    }

print(f"\n✓ Phase 2 complete for {len(phase2_results)} admissions")

with_alt_causes  = sum(1 for v in phase2_results.values() if v["alternative_causes_found"])
with_codrugs     = sum(1 for v in phase2_results.values() if v["other_nephrotoxic_drugs"])
missing_diagnosis = sum(1 for v in phase2_results.values() if not v["diagnosis_text_available"])

print(f"\nPhase 2 Summary:")
print(f"  Admissions with alternative causes found    : {with_alt_causes}")
print(f"  Admissions with other nephrotoxic co-drugs   : {with_codrugs}")
print(f"  Admissions with no diagnosis data available  : {missing_diagnosis}")


# ============================================================
# PHASE 3 — LIGHTWEIGHT NLP ON CLINICAL NOTES
# ============================================================

print("\n" + "=" * 55)
print("PHASE 3 — Lightweight NLP on Clinical Notes")
print("=" * 55)

print("\nLoading clinical notes...")
notes = pd.read_csv(
    os.path.join(FOLDER, "discharge.csv clinical notes of MiMIC IV.gz"),
    compression="gzip",
    low_memory=False
)
print(f"  {len(notes):,} notes loaded")

SEARCH_TERMS = [
    "aki", "renal", "creatinine", "vancomycin",
    "nephrotoxicity", "toxicity", "held", "renal impairment"
]

NEGATION_PHRASES = [
    "no aki",
    "not due to vancomycin",
    "unlikely related",
]

POSITIVE_EVIDENCE_PHRASES = [
    "aki likely multifactorial",
    "vancomycin held due to renal impairment",
    "renal function worsening",
    "possible nephrotoxicity",
    "sepsis-related aki",
]


def extract_note_evidence(note_text):
    """
    Pull out sentences containing any search term.
    Classify as negated / positive evidence / needs manual review.
    Does not attempt a final causality decision.
    """
    if not isinstance(note_text, str) or len(note_text) < 10:
        return []

    findings = []
    sentences = re.split(r'[.\n;]+', note_text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue

        sentence_lower = sentence.lower()

        matched_terms = [term for term in SEARCH_TERMS if term in sentence_lower]
        if not matched_terms:
            continue

        negation_found = next(
            (phrase for phrase in NEGATION_PHRASES if phrase in sentence_lower), None
        )
        positive_found = next(
            (phrase for phrase in POSITIVE_EVIDENCE_PHRASES if phrase in sentence_lower), None
        )

        if negation_found:
            classification = f"negation found ('{negation_found}')"
        elif positive_found:
            classification = f"positive evidence phrase found ('{positive_found}')"
        else:
            classification = "mentions search term — needs manual review"

        findings.append({
            "sentence": sentence[:300],
            "matched_terms": matched_terms,
            "classification": classification,
        })

    return findings


def get_note_evidence_for_admission(hadm_id, notes_df):
    patient_notes = notes_df[notes_df["hadm_id"] == hadm_id]

    if patient_notes.empty:
        return {"note_evidence": [], "notes_available": False}

    all_findings = []
    for _, note_row in patient_notes.iterrows():
        findings = extract_note_evidence(note_row.get("text", ""))
        all_findings.extend(findings)

    return {"note_evidence": all_findings, "notes_available": True}


print("\nExtracting note evidence for 20 admissions...\n")

phase3_results = {}

for _, row in sample_timing.iterrows():
    hadm_id = row["hadm_id"]
    print(f"  Processing admission {hadm_id}...")
    phase3_results[hadm_id] = get_note_evidence_for_admission(hadm_id, notes)

print(f"\n Phase 3 complete for {len(phase3_results)} admissions")

with_notes       = sum(1 for v in phase3_results.values() if v["notes_available"])
with_evidence    = sum(1 for v in phase3_results.values() if v["note_evidence"])
total_sentences  = sum(len(v["note_evidence"]) for v in phase3_results.values())

print(f"\nPhase 3 Summary:")
print(f"  Admissions with notes available         : {with_notes}")
print(f"  Admissions with relevant sentences found : {with_evidence}")
print(f"  Total relevant sentences extracted       : {total_sentences}")


# ============================================================
# COMBINATION OF PHASES 1, 2, 3 INTO ONE FINAL TABLE
# ============================================================

print("\n" + "=" * 55)
print("COMBINING ALL THREE PHASES")
print("=" * 55)

combined_results = []

for r in results:
    hadm_id = r["hadm_id"]

    p2 = phase2_results.get(hadm_id, {})
    p3 = phase3_results.get(hadm_id, {})

    missing = r["missing_evidence"].split("; ") if r["missing_evidence"] != "none" else []

    if not p2.get("diagnosis_text_available", False):
        missing.append("diagnosis codes unavailable for alternative-cause check")
    if not p3.get("notes_available", False):
        missing.append("no clinical notes found for this admission")

    combined_results.append({
        "subject_id": r["subject_id"],
        "hadm_id": hadm_id,
        "suspected_drug": r["suspected_drug"],
        "drug_start_time": r["drug_start_time"],
        "drug_stop_time": r["drug_stop_time"],
        "baseline_creatinine": r["baseline_creatinine"],
        "peak_creatinine": r["peak_creatinine"],
        "aki_signal": r["aki_signal"],
        "temporal_relationship": r["temporal_relationship"],
        "dechallenge": r["dechallenge"],
        "possible_alternative_causes": p2.get("alternative_causes_found", []),
        "other_nephrotoxic_drugs": p2.get("other_nephrotoxic_drugs", []),
        "relevant_note_evidence": p3.get("note_evidence", [])[:10],
        "missing_evidence": "; ".join(missing) if missing else "none",
    })

print(f" Combined {len(combined_results)} admissions across all 3 phases")


# ============================================================
# SAVE FINAL OUTPUTS
# ============================================================

json_path = os.path.join(FOLDER, "vancomycin_aki_full_evidence.json")
with open(json_path, "w") as f:
    json.dump(combined_results, f, indent=2, default=str)
print(f"\n✓ Full detail JSON saved:\n  {json_path}")

summary_rows = []
for r in combined_results:
    summary_rows.append({
        "subject_id": r["subject_id"],
        "hadm_id": r["hadm_id"],
        "suspected_drug": r["suspected_drug"],
        "drug_start_time": r["drug_start_time"],
        "drug_stop_time": r["drug_stop_time"],
        "baseline_creatinine": r["baseline_creatinine"],
        "peak_creatinine": r["peak_creatinine"],
        "aki_signal": r["aki_signal"],
        "temporal_relationship": r["temporal_relationship"],
        "dechallenge": r["dechallenge"],
        "alternative_causes": "; ".join(r["possible_alternative_causes"]) or "none found",
        "other_nephrotoxic_drugs": "; ".join(r["other_nephrotoxic_drugs"]) or "none found",
        "note_sentences_found": len(r["relevant_note_evidence"]),
        "top_note_sentence": (
            r["relevant_note_evidence"][0]["sentence"] if r["relevant_note_evidence"] else "none found"
        ),
        "missing_evidence": r["missing_evidence"],
    })

summary_df = pd.DataFrame(summary_rows)

csv_path   = os.path.join(FOLDER, "vancomycin_aki_full_summary.csv")
excel_path = os.path.join(FOLDER, "vancomycin_aki_full_summary.xlsx")

summary_df.to_csv(csv_path, index=False)
summary_df.to_excel(excel_path, index=False)

print(f"✓ Summary CSV saved:\n  {csv_path}")
print(f"✓ Summary Excel saved:\n  {excel_path}")


# ============================================================
# FINAL SUMMARY — ALL THREE PHASES
# ============================================================

total = len(combined_results)

print("\n" + "=" * 55)
print("FULL PIPELINE SUMMARY — PHASES 1, 2, 3")
print("=" * 55)
print(f"Admissions processed                      : {total}")
print()
print("PHASE 1 — Structured Evidence:")
print(f"  AKI signal YES      : {sum(1 for r in combined_results if 'YES' in str(r['aki_signal']))}")
print(f"  AKI signal NO       : {sum(1 for r in combined_results if 'NO' in str(r['aki_signal']))}")
print(f"  AKI signal unknown  : {sum(1 for r in combined_results if 'unknown' in str(r['aki_signal']))}")
print(f"  Dechallenge positive: {sum(1 for r in combined_results if 'positive' in str(r['dechallenge']))}")
print()
print("PHASE 2 — Alternative Causes:")
print(f"  Cases with alternative causes found  : {sum(1 for r in combined_results if r['possible_alternative_causes'])}")
print(f"  Cases with other nephrotoxic co-drugs: {sum(1 for r in combined_results if r['other_nephrotoxic_drugs'])}")
print()
print("PHASE 3 — Note Evidence:")
print(f"  Cases with relevant note sentences found: {sum(1 for r in combined_results if r['relevant_note_evidence'])}")
print(f"  Total note sentences extracted          : {sum(len(r['relevant_note_evidence']) for r in combined_results)}")
print()
print(f"Cases with any missing evidence flagged: {sum(1 for r in combined_results if r['missing_evidence'] != 'none')}")
print("=" * 55)
print("\nAll three phases complete. Ready for manual review.")


for r in combined_results:
    if r["missing_evidence"] != "none":
        print(f"{r['hadm_id']}: {r['missing_evidence']}")
        
        
        
        
        
# Quick sanity check — are the "missing notes" cases short admissions?
for hadm_id in [23280645, 25860671, 23705591, 29250371, 26133978, 25614151, 27923846]:
    row = sample_timing[sample_timing["hadm_id"] == hadm_id]
    print(f"{hadm_id}: doses={row['total_doses'].values[0]}, "
          f"start={row['drug_start_time'].values[0]}, stop={row['drug_stop_time'].values[0]}")# Quick sanity check — are the "missing notes" cases short admissions?
for hadm_id in [23280645, 25860671, 23705591, 29250371, 26133978, 25614151, 27923846]:
    row = sample_timing[sample_timing["hadm_id"] == hadm_id]
    print(f"{hadm_id}: doses={row['total_doses'].values[0]}, "
          f"start={row['drug_start_time'].values[0]}, stop={row['drug_stop_time'].values[0]}")

    
for hadm_id in [21334040, 24906418, 27703517, 27012892, 28185499]:
    n = creatinine[creatinine["hadm_id"] == hadm_id].shape[0]
    print(f"{hadm_id}: {n} total creatinine measurements found")
    
    
# Check 1 — does this discharge.csv only cover a subset of all admissions,
# regardless of Vancomycin status?
total_admissions_with_notes = notes["hadm_id"].nunique()
total_admissions_overall = admissions["hadm_id"].nunique()
print(f"Admissions with at least one note: {total_admissions_with_notes:,} of {total_admissions_overall:,}")
print(f"Coverage: {total_admissions_with_notes/total_admissions_overall*100:.1f}%")


# Check 2 — does this specific patient have ANY notes anywhere,
# just possibly under a different hadm_id or as a different note type?
for hadm_id in [25860671, 27923846]:
    subj = sample_timing[sample_timing["hadm_id"] == hadm_id]["subject_id"].values[0]
    other_notes = notes[notes["subject_id"] == subj] if "subject_id" in notes.columns else pd.DataFrame()
    print(f"hadm_id={hadm_id}, subject_id={subj}: {len(other_notes)} notes for this patient across ALL admissions")
    
# Check exactly how far back the nearest pre-drug creatinine actually is
for hadm_id in [21334040, 24906418, 27703517, 27012892, 28185499]:
    drug_start = sample_timing[sample_timing["hadm_id"] == hadm_id]["drug_start_time"].values[0]
    patient_creat = creatinine[creatinine["hadm_id"] == hadm_id].sort_values("charttime")
    pre_drug = patient_creat[patient_creat["charttime"] < drug_start]
    if not pre_drug.empty:
        nearest = pre_drug.iloc[-1]
        hours_before = (pd.to_datetime(drug_start) - nearest["charttime"]).total_seconds() / 3600
        print(f"{hadm_id}: nearest pre-drug creatinine was {hours_before:.0f} hours before drug start "
              f"(value={nearest['valuenum']})")
    else:
        print(f"{hadm_id}: NO creatinine at all before drug start (all {len(patient_creat)} readings are AFTER)")
        
        
        