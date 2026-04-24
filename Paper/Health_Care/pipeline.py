#!/usr/bin/env python3
"""
restructure_for_pipeline.py
════════════════════════════════════════════════════════════════════════
Restructures the filtered MIMIC-IV mental health datasets into a clean,
purpose-built dataset for the three-stage pipeline:

  Stage 1: Clinical Factor Extraction
  Stage 2: Extreme Scenario Generation (Counterfactual Simulation)
  Stage 3: Causal Explainability (XAI)

Operations:
  ✓ Select only columns needed for the pipeline
  ✓ Filter lab/chart events to mental-health-relevant items
  ✓ Drop unnecessary tables (emar_detail, poe, poe_detail, pharmacy, etc.)
  ✓ Output clean CSV files with clear naming

Input:
  Already-filtered mental health subsets from:
    Dataset1/.../Output/hosp_mental_health/
    Dataset1/.../Output/icu_mental_health/
    Dataset2/.../output/

Output:
  Pipeline_Data/  (restructured CSVs ready for pipeline implementation)

Tables DROPPED (not needed for pipeline):
  - emar_detail    (61M rows — dose detail; emar already has medication + event)
  - poe            (24M rows — provider order entry; redundant with prescriptions)
  - poe_detail     (5.5M rows — order detail)
  - pharmacy       (8.4M rows — largely redundant with prescriptions)
  - hcpcsevents    (billing events; not clinically useful)
  - d_hcpcs        (reference for above)
  - provider       (just provider IDs, no useful attributes)
  - caregiver      (just caregiver IDs, no useful attributes)
  - microbiologyevents (culture results; not central to MH pipeline)
  - datetimeevents (ICU datetime stamps; not critical)
  - ingredientevents (IV ingredient details; too granular)

Estimated run time: 30-60 minutes (depends on I/O speed)
════════════════════════════════════════════════════════════════════════
"""

import csv
import os
import sys
import time
import re

csv.field_size_limit(sys.maxsize)

# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = "/media/sandisk/Ai_Drive"

HOSP_MH = os.path.join(BASE_DIR, "Dataset1/mimic-iv-3.1/Output/hosp_mental_health")
ICU_MH  = os.path.join(BASE_DIR, "Dataset1/mimic-iv-3.1/Output/icu_mental_health")
NOTES   = os.path.join(BASE_DIR, "Dataset2/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/output")
MH_ICD  = os.path.join(BASE_DIR, "Dataset1/mimic-iv-3.1/Output/d_icd_diagnoses_mental_health.csv")

OUT_DIR = os.path.join(BASE_DIR, "Pipeline_Data")
REF_DIR = os.path.join(OUT_DIR, "reference")


# ═════════════════════════════════════════════════════════════════════════════
# MENTAL-HEALTH-RELEVANT LAB KEYWORDS
# Matched against d_labitems.label to discover relevant itemids
# ═════════════════════════════════════════════════════════════════════════════

MH_LAB_KEYWORDS = [
    # Drug levels / therapeutic monitoring
    r"lithium", r"valproic", r"valproate", r"carbamazepine",
    r"phenytoin", r"clozapine", r"lamotrigine", r"phenobarbital",
    # Toxicology / drug screens
    r"drug screen", r"toxicology", r"alcohol", r"ethanol",
    r"acetaminophen", r"salicylate", r"benzodiazepine", r"barbiturate",
    r"opiate", r"opioid", r"cocaine", r"amphetamine", r"cannabin",
    r"methadone", r"phencyclidine",
    # Thyroid
    r"thyroid", r"\bTSH\b", r"\bT3\b", r"\bT4\b", r"thyroxine",
    r"free T4", r"triiodothyronine",
    # Metabolic
    r"glucose", r"hemoglobin A1c", r"HbA1c", r"\bA1c\b",
    # Kidney (lithium monitoring)
    r"creatinine", r"\bBUN\b", r"\bGFR\b", r"urea nitrogen",
    # Hematology (clozapine monitoring)
    r"\bWBC\b", r"white blood", r"neutrophil", r"\bANC\b",
    r"platelet", r"hematocrit", r"hemoglobin",
    # Liver (mood stabilizer monitoring)
    r"\bAST\b", r"\bALT\b", r"bilirubin", r"alkaline phosphatase",
    r"alanine aminotransferase", r"aspartate aminotransferase",
    # Metabolic syndrome (antipsychotic monitoring)
    r"triglyceride", r"cholesterol", r"\bHDL\b", r"\bLDL\b", r"prolactin",
    # Electrolytes
    r"sodium", r"potassium", r"magnesium", r"calcium", r"phosph",
    r"bicarbonate", r"\bCO2\b", r"chloride",
    # Other MH-relevant
    r"ammonia",                                # valproic acid toxicity
    r"\bCPK\b", r"\bCK\b", r"creatine kinase", # neuroleptic malignant syndrome
    r"cortisol",
    r"vitamin B12", r"folate", r"folic acid",
    r"\bCRP\b", r"C-reactive", r"\bESR\b",     # inflammatory markers
    r"iron", r"ferritin", r"transferrin",
    r"albumin",
    r"lactate",
    # Blood gas basics (overdose/withdrawal)
    r"\bpH\b", r"\bpCO2\b", r"\bpO2\b",
    r"oxygen saturation",
]

# ═════════════════════════════════════════════════════════════════════════════
# MENTAL-HEALTH-RELEVANT ICU CHART/INPUT ITEM KEYWORDS
# Matched against d_items.label to discover relevant itemids
# ═════════════════════════════════════════════════════════════════════════════

MH_ICU_KEYWORDS = [
    # Mental status assessment
    r"\bGCS\b", r"Glasgow", r"consciousness",
    r"\bRASS\b", r"Richmond", r"agitation",
    r"\bCAM\b", r"delirium", r"confusion",
    r"orientation", r"orient",
    r"pupil",
    # Safety / psych-specific
    r"restraint", r"suicide", r"safety", r"precaution",
    r"fall risk", r"sitter", r"1:1",
    r"behavior", r"psych",
    # Pain
    r"pain",
    # Vital signs (all relevant — autonomic instability, withdrawal, serotonin syndrome)
    r"heart rate",
    r"blood pressure", r"systolic", r"diastolic",
    r"respiratory rate",
    r"O2 sat", r"SpO2", r"pulse ox", r"oxygen sat",
    r"temperature",
    # Non-invasive BP (specific item names)
    r"\bNBP\b", r"\bABP\b",
    # Weight
    r"weight", r"\bBMI\b",
    # Neurological
    r"neuro",
    # Common MH IV medications (for inputevents)
    r"haloperidol", r"lorazepam", r"midazolam", r"diazepam",
    r"olanzapine", r"ziprasidone", r"ketamine",
    r"dexmedetomidine", r"propofol",
    r"phenobarbital", r"valproic", r"levetiracetam",
    r"naloxone", r"flumazenil",
    r"morphine", r"fentanyl", r"hydromorphone",
    # Sedation
    r"sedation", r"sedative",
]


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def discover_relevant_itemids(filepath, keywords, label_col="label"):
    """
    Scan a dictionary table (d_labitems or d_items) and return
    the set of itemids whose label matches any keyword pattern.
    """
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    itemids = set()
    matched = {}
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get(label_col, "")
            if pattern.search(label):
                iid = row["itemid"].strip()
                itemids.add(iid)
                matched[iid] = label.strip()
    return itemids, matched


def process_table(src_path, dst_path, columns_to_keep,
                  row_filter=None, progress_every=1_000_000,
                  description=""):
    """
    Stream-process a CSV file:
      - Select only specified columns (or all if columns_to_keep is None)
      - Optionally filter rows via row_filter(row_as_list, col_index_map) -> bool
      - Print progress for large files

    Uses csv.reader (not DictReader) for speed on large files.
    Returns (rows_kept, rows_scanned).
    """
    if not os.path.exists(src_path):
        print(f"  ⚠  SKIP (file not found): {os.path.basename(src_path)}")
        return 0, 0

    t0 = time.time()
    kept = 0
    total = 0

    with open(src_path, "r", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)

        # Build column index map
        col_idx = {name: i for i, name in enumerate(header)}

        # Determine which columns to write
        if columns_to_keep is None:
            out_cols = header
            out_indices = list(range(len(header)))
        else:
            out_cols = [c for c in columns_to_keep if c in col_idx]
            out_indices = [col_idx[c] for c in out_cols]

        with open(dst_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(out_cols)

            for row in reader:
                total += 1
                if row_filter is None or row_filter(row, col_idx):
                    writer.writerow([row[i] for i in out_indices])
                    kept += 1

                if total % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed if elapsed > 0 else 0
                    pct = (kept / total * 100) if total > 0 else 0
                    print(f"    ... {total:>14,} scanned | {kept:>12,} kept "
                          f"({pct:.1f}%) | {elapsed:.0f}s | "
                          f"{rate:,.0f} rows/s")

    elapsed = time.time() - t0
    desc = description or os.path.basename(dst_path)
    print(f"  ✓ {desc:<30s}  {kept:>14,} rows  ({elapsed:.1f}s)")
    return kept, total


def save_filtered_reference(src_path, dst_path, itemids, description=""):
    """Save only the rows from a reference table whose itemid is in the set."""
    if not os.path.exists(src_path):
        print(f"  ⚠  SKIP (not found): {os.path.basename(src_path)}")
        return 0

    kept = 0
    with open(src_path, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames

        with open(dst_path, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if row["itemid"].strip() in itemids:
                    writer.writerow(row)
                    kept += 1

    desc = description or os.path.basename(dst_path)
    print(f"  ✓ {desc:<30s}  {kept:>14,} rows")
    return kept


def copy_file(src_path, dst_path, description=""):
    """Copy a file as-is (for small reference tables)."""
    if not os.path.exists(src_path):
        print(f"  ⚠  SKIP (not found): {os.path.basename(src_path)}")
        return

    import shutil
    shutil.copy2(src_path, dst_path)
    # Count rows
    with open(dst_path, "r") as f:
        n = sum(1 for _ in f) - 1  # exclude header
    desc = description or os.path.basename(dst_path)
    print(f"  ✓ {desc:<30s}  {n:>14,} rows  (copied)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  MIMIC-IV Mental Health → Pipeline Dataset Restructuring")
    print("=" * 72)
    print(f"  Output directory: {OUT_DIR}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)

    overall_t0 = time.time()
    summary = {}

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 0: Discover MH-relevant item IDs for filtering large tables
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 0: Discovering MH-relevant item IDs ──────────────────")

    lab_itemids, lab_labels = discover_relevant_itemids(
        os.path.join(HOSP_MH, "d_labitems.csv"), MH_LAB_KEYWORDS
    )
    print(f"  Lab items matched: {len(lab_itemids)} / 1650")

    icu_itemids, icu_labels = discover_relevant_itemids(
        os.path.join(ICU_MH, "d_items.csv"), MH_ICU_KEYWORDS
    )
    print(f"  ICU items matched: {len(icu_itemids)} / 4095")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: Hospital tables — small/medium (column selection only)
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 1: Hospital tables (column selection) ────────────────")

    # 1. patients
    n, _ = process_table(
        os.path.join(HOSP_MH, "patients.csv"),
        os.path.join(OUT_DIR, "patients.csv"),
        ["subject_id", "gender", "anchor_age", "anchor_year",
         "anchor_year_group", "dod"],
        description="patients"
    )
    summary["patients"] = n

    # 2. admissions
    n, _ = process_table(
        os.path.join(HOSP_MH, "admissions.csv"),
        os.path.join(OUT_DIR, "admissions.csv"),
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
         "admission_type", "admission_location", "discharge_location",
         "insurance", "marital_status", "race", "hospital_expire_flag"],
        description="admissions"
    )
    summary["admissions"] = n

    # 3. diagnoses_icd (keep all columns — all needed)
    n, _ = process_table(
        os.path.join(HOSP_MH, "diagnoses_icd.csv"),
        os.path.join(OUT_DIR, "diagnoses_icd.csv"),
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        description="diagnoses_icd"
    )
    summary["diagnoses_icd"] = n

    # 4. services
    n, _ = process_table(
        os.path.join(HOSP_MH, "services.csv"),
        os.path.join(OUT_DIR, "services.csv"),
        ["subject_id", "hadm_id", "transfertime", "prev_service",
         "curr_service"],
        description="services"
    )
    summary["services"] = n

    # 5. transfers
    n, _ = process_table(
        os.path.join(HOSP_MH, "transfers.csv"),
        os.path.join(OUT_DIR, "transfers.csv"),
        ["subject_id", "hadm_id", "eventtype", "careunit",
         "intime", "outtime"],
        description="transfers"
    )
    summary["transfers"] = n

    # 6. drgcodes (DRG severity scores)
    n, _ = process_table(
        os.path.join(HOSP_MH, "drgcodes.csv"),
        os.path.join(OUT_DIR, "drg_severity.csv"),
        ["subject_id", "hadm_id", "drg_type", "drg_code",
         "description", "drg_severity", "drg_mortality"],
        description="drg_severity"
    )
    summary["drg_severity"] = n

    # 7. procedures_icd
    n, _ = process_table(
        os.path.join(HOSP_MH, "procedures_icd.csv"),
        os.path.join(OUT_DIR, "procedures_icd.csv"),
        ["subject_id", "hadm_id", "seq_num", "chartdate",
         "icd_code", "icd_version"],
        description="procedures_icd"
    )
    summary["procedures_icd"] = n

    # 8. omr (outpatient vitals — BMI, BP)
    n, _ = process_table(
        os.path.join(HOSP_MH, "omr.csv"),
        os.path.join(OUT_DIR, "vitals_omr.csv"),
        ["subject_id", "chartdate", "result_name", "result_value"],
        description="vitals_omr"
    )
    summary["vitals_omr"] = n

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: Hospital tables — large (column selection + filtering)
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 2: Large hospital tables (select + filter) ───────────")

    # 9. prescriptions (9.5M rows — keep all rows, drop non-essential columns)
    n, _ = process_table(
        os.path.join(HOSP_MH, "prescriptions.csv"),
        os.path.join(OUT_DIR, "prescriptions.csv"),
        ["subject_id", "hadm_id", "starttime", "stoptime",
         "drug_type", "drug", "prod_strength",
         "dose_val_rx", "dose_unit_rx", "route", "doses_per_24_hrs"],
        description="prescriptions"
    )
    summary["prescriptions"] = n

    # 10. emar — medication administration records (22.5M rows)
    #     Keep only key columns for adherence tracking
    n, _ = process_table(
        os.path.join(HOSP_MH, "emar.csv"),
        os.path.join(OUT_DIR, "medication_admin.csv"),
        ["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
        description="medication_admin (emar)"
    )
    summary["medication_admin"] = n

    # 11. labevents — FILTERED to MH-relevant lab items only (40M → ~10-15M)
    def lab_filter(row, col_idx):
        return row[col_idx["itemid"]].strip() in lab_itemids

    n, total = process_table(
        os.path.join(HOSP_MH, "labevents.csv"),
        os.path.join(OUT_DIR, "lab_results.csv"),
        ["subject_id", "hadm_id", "itemid", "charttime",
         "value", "valuenum", "valueuom",
         "ref_range_lower", "ref_range_upper", "flag"],
        row_filter=lab_filter,
        description="lab_results (filtered)"
    )
    summary["lab_results"] = n
    print(f"           (kept {n:,} / {total:,} = "
          f"{n/total*100:.1f}% of lab events)")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3: ICU tables
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 3: ICU tables ────────────────────────────────────────")

    # 12. icustays (small — 45K rows)
    n, _ = process_table(
        os.path.join(ICU_MH, "icustays.csv"),
        os.path.join(OUT_DIR, "icu_stays.csv"),
        ["subject_id", "hadm_id", "stay_id", "first_careunit",
         "last_careunit", "intime", "outtime", "los"],
        description="icu_stays"
    )
    summary["icu_stays"] = n

    # 13. chartevents — FILTERED to MH-relevant items (222M → ~30-60M)
    def chart_filter(row, col_idx):
        return row[col_idx["itemid"]].strip() in icu_itemids

    n, total = process_table(
        os.path.join(ICU_MH, "chartevents.csv"),
        os.path.join(OUT_DIR, "icu_chart_events.csv"),
        ["subject_id", "hadm_id", "stay_id", "charttime",
         "itemid", "value", "valuenum", "valueuom"],
        row_filter=chart_filter,
        description="icu_chart_events (filtered)"
    )
    summary["icu_chart_events"] = n
    if total > 0:
        print(f"           (kept {n:,} / {total:,} = "
              f"{n/total*100:.1f}% of chart events)")

    # 14. inputevents — FILTERED to MH-relevant items (5.5M → ~1-2M)
    def input_filter(row, col_idx):
        return row[col_idx["itemid"]].strip() in icu_itemids

    n, total = process_table(
        os.path.join(ICU_MH, "inputevents.csv"),
        os.path.join(OUT_DIR, "icu_input_events.csv"),
        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime",
         "itemid", "amount", "amountuom", "rate", "rateuom",
         "ordercategorydescription", "statusdescription"],
        row_filter=input_filter,
        description="icu_input_events (filtered)"
    )
    summary["icu_input_events"] = n
    if total > 0:
        print(f"           (kept {n:,} / {total:,} = "
              f"{n/total*100:.1f}% of input events)")

    # 15. outputevents (2.6M rows — keep all, select columns)
    n, _ = process_table(
        os.path.join(ICU_MH, "outputevents.csv"),
        os.path.join(OUT_DIR, "icu_output_events.csv"),
        ["subject_id", "hadm_id", "stay_id", "charttime",
         "itemid", "value", "valueuom"],
        description="icu_output_events"
    )
    summary["icu_output_events"] = n

    # 16. procedureevents (384K rows — keep all, select columns)
    n, _ = process_table(
        os.path.join(ICU_MH, "procedureevents.csv"),
        os.path.join(OUT_DIR, "icu_procedure_events.csv"),
        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime",
         "itemid", "value", "valueuom", "statusdescription"],
        description="icu_procedure_events"
    )
    summary["icu_procedure_events"] = n

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 4: Clinical notes (Dataset2)
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 4: Clinical notes ────────────────────────────────────")

    # 17. Discharge notes (the PRIMARY input to the LLM)
    n, _ = process_table(
        os.path.join(NOTES, "discharge_mental_health.csv"),
        os.path.join(OUT_DIR, "discharge_notes.csv"),
        ["note_id", "subject_id", "hadm_id", "charttime", "text"],
        description="discharge_notes"
    )
    summary["discharge_notes"] = n

    # 18. Radiology notes (brain imaging findings)
    n, _ = process_table(
        os.path.join(NOTES, "radiology_mental_health.csv"),
        os.path.join(OUT_DIR, "radiology_notes.csv"),
        ["note_id", "subject_id", "hadm_id", "charttime", "text"],
        description="radiology_notes"
    )
    summary["radiology_notes"] = n

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 5: Reference tables
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 5: Reference tables ──────────────────────────────────")

    # MH ICD codes (pre-filtered reference)
    copy_file(MH_ICD,
              os.path.join(REF_DIR, "mh_icd_codes.csv"),
              "mh_icd_codes")

    # Full ICD diagnoses dictionary (needed for comorbidity lookup)
    copy_file(os.path.join(HOSP_MH, "d_icd_diagnoses.csv"),
              os.path.join(REF_DIR, "d_icd_diagnoses.csv"),
              "d_icd_diagnoses (full)")

    # Full ICD procedures dictionary
    copy_file(os.path.join(HOSP_MH, "d_icd_procedures.csv"),
              os.path.join(REF_DIR, "d_icd_procedures.csv"),
              "d_icd_procedures (full)")

    # Filtered lab items reference (only MH-relevant items)
    save_filtered_reference(
        os.path.join(HOSP_MH, "d_labitems.csv"),
        os.path.join(REF_DIR, "lab_items.csv"),
        lab_itemids,
        "lab_items (MH-relevant)"
    )

    # Filtered ICU items reference (only MH-relevant items)
    save_filtered_reference(
        os.path.join(ICU_MH, "d_items.csv"),
        os.path.join(REF_DIR, "icu_items.csv"),
        icu_itemids,
        "icu_items (MH-relevant)"
    )

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 6: Generate schema reference
    # ─────────────────────────────────────────────────────────────────────
    print("\n─── Phase 6: Generating schema reference ───────────────────────")

    schema_text = generate_schema_reference(summary)
    schema_path = os.path.join(OUT_DIR, "SCHEMA_REFERENCE.txt")
    with open(schema_path, "w") as f:
        f.write(schema_text)
    print(f"  ✓ Schema written to {schema_path}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    elapsed = time.time() - overall_t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 72)
    print(f"  RESTRUCTURING COMPLETE  ({minutes}m {seconds}s)")
    print("=" * 72)
    print(f"\n  Output: {OUT_DIR}")
    print(f"\n  {'Table':<30s}  {'Rows':>14s}")
    print(f"  {'─' * 30}  {'─' * 14}")
    for name, count in summary.items():
        print(f"  {name:<30s}  {count:>14,}")
    total_rows = sum(summary.values())
    print(f"  {'─' * 30}  {'─' * 14}")
    print(f"  {'TOTAL':<30s}  {total_rows:>14,}")

    # Show output directory size
    total_size = 0
    for dirpath, _, filenames in os.walk(OUT_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    print(f"\n  Total output size: {total_size / (1024**3):.2f} GB")
    print()


def generate_schema_reference(summary):
    """Generate a human-readable schema reference for the output dataset."""
    return f"""\
================================================================================
        Pipeline Dataset — Schema Reference
        Restructured from MIMIC-IV v3.1 + MIMIC-IV-Note v2.2
        Mental Health Cohort (ICD-9: 290-319, ICD-10: F01-F99)
================================================================================


TABLES OVERVIEW
───────────────
{chr(10).join(f'  {name:<30s}  {count:>14,} rows' for name, count in summary.items())}


════════════════════════════════════════════════════════════════════════════════
                         TABLE DEFINITIONS
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
1. patients.csv
   Purpose: Patient demographics — age, gender, mortality
   Stage:   1 (Clinical Factor Extraction)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         PK   Unique patient identifier
   gender               VARCHAR          Patient gender (M/F)
   anchor_age           INT              Age at anchor_year
   anchor_year          INT              Shifted year for de-identification
   anchor_year_group    VARCHAR          Year range group (e.g. "2014 - 2016")
   dod                  DATE             Date of death (if applicable)

   Key uses:
   • Demographics for clinical factor profile
   • dod → mortality outcome (Stage 2 extreme scenario labeling)
   • anchor_age → age-based risk stratification

────────────────────────────────────────────────────────────────────────────────
2. admissions.csv
   Purpose: Hospital admission context — type, location, outcome
   Stage:   1, 2 (admission trajectories, readmission detection)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         PK   Unique hospital admission ID
   admittime            DATETIME         Admission date/time
   dischtime            DATETIME         Discharge date/time
   deathtime            DATETIME         In-hospital death time (if applicable)
   admission_type       VARCHAR          Type (EMERGENCY, URGENT, ELECTIVE, etc.)
   admission_location   VARCHAR          Location prior to admission
   discharge_location   VARCHAR          Discharge destination
   insurance            VARCHAR          Insurance type (resource access proxy)
   marital_status       VARCHAR          Marital status (isolation proxy)
   race                 VARCHAR          Patient race
   hospital_expire_flag INT              1 = died during this admission

   Key uses:
   • admission_type="EMERGENCY" → acute decompensation signal
   • discharge_location="PSYCH" → psychiatric disposition
   • hospital_expire_flag → in-hospital mortality (Stage 2 extreme outcome)
   • admittime ordering → readmission interval calculation
   • marital_status → social isolation risk factor

────────────────────────────────────────────────────────────────────────────────
3. diagnoses_icd.csv
   Purpose: All ICD diagnoses per admission (MH + comorbidities)
   Stage:   1, 2, 3 (diagnosis extraction, trajectory comparison, causal features)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   seq_num              INT              Diagnosis priority (1 = primary)
   icd_code             VARCHAR     FK → reference/d_icd_diagnoses
   icd_version          INT              ICD version (9 or 10)

   Key uses:
   • Primary MH diagnosis identification (seq_num=1 or first MH code)
   • Comorbidity profiling (non-MH codes)
   • Cross-admission diagnosis escalation (Stage 2 trajectory pairs)
   • Join with reference/mh_icd_codes.csv to identify MH-specific codes

────────────────────────────────────────────────────────────────────────────────
4. prescriptions.csv
   Purpose: Medication orders — what was prescribed
   Stage:   1, 2, 3 (medication profile, trajectory changes, causal features)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   starttime            DATETIME         Prescription start
   stoptime             DATETIME         Prescription stop
   drug_type            VARCHAR          MAIN / BASE / ADDITIVE
   drug                 VARCHAR          Drug name
   prod_strength        VARCHAR          Product strength (e.g. "40mg Tablet")
   dose_val_rx          VARCHAR          Dose value prescribed
   dose_unit_rx         VARCHAR          Dose unit (mg, mL, etc.)
   route                VARCHAR          Route (PO, IV, IM, etc.)
   doses_per_24_hrs     FLOAT            Doses per 24 hours

   Key uses:
   • Psychotropic medication identification (SSRIs, antipsychotics, etc.)
   • Dose escalation/de-escalation between admissions
   • Polypharmacy count
   • Route change (PO→IV = severity escalation)
   • Drug class changes across admissions (Stage 2 trajectories)

   Columns DROPPED (not needed):
   pharmacy_id, poe_id, poe_seq, order_provider_id,
   formulary_drug_cd, gsn, ndc, form_rx, form_val_disp, form_unit_disp

────────────────────────────────────────────────────────────────────────────────
5. medication_admin.csv   (source: emar)
   Purpose: Actual medication administration — adherence signals
   Stage:   1, 3 (adherence extraction, causal risk factor)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   charttime            DATETIME         Administration time
   medication           VARCHAR          Medication name
   event_txt            VARCHAR          Event outcome:
                                           Administered, Not Given, Hold Dose,
                                           Refused, Stopped, Rate Change, etc.

   Key uses:
   • event_txt="Not Given" or "Hold Dose" → non-adherence signal
   • Frequency of "Administered" → medication compliance rate
   • Temporal patterns (refused at night vs. day)
   • Critical for Stage 3 causal chain: non-adherence → decompensation

   Columns DROPPED (not needed):
   emar_id, emar_seq, poe_id, pharmacy_id, enter_provider_id,
   scheduletime, storetime

────────────────────────────────────────────────────────────────────────────────
6. lab_results.csv   (source: labevents, FILTERED to MH-relevant items)
   Purpose: Lab results relevant to mental health monitoring
   Stage:   1, 2, 3 (biomarker extraction, severity signals, causal features)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   itemid               INT         FK → reference/lab_items.csv
   charttime            DATETIME         Time of result
   value                VARCHAR          Result (text)
   valuenum             FLOAT            Result (numeric)
   valueuom             VARCHAR          Unit of measure
   ref_range_lower      FLOAT            Normal range lower bound
   ref_range_upper      FLOAT            Normal range upper bound
   flag                 VARCHAR          "abnormal" if out of range

   Included lab categories:
   • Drug levels: Lithium, Valproic acid, Carbamazepine, Phenytoin
   • Toxicology: Urine drug screens, blood alcohol, acetaminophen
   • Thyroid: TSH, T3, T4 (hypothyroidism mimics depression)
   • Metabolic: Glucose, HbA1c (antipsychotic metabolic syndrome)
   • Kidney: Creatinine, BUN (lithium monitoring)
   • Hematology: WBC, ANC (clozapine monitoring)
   • Liver: AST, ALT, bilirubin (mood stabilizer hepatotoxicity)
   • Lipids: Triglycerides, cholesterol (antipsychotic side effects)
   • Other: CPK/CK (neuroleptic malignant syndrome), electrolytes,
     ammonia (valproic acid toxicity), cortisol, B12/folate

   Columns DROPPED: labevent_id, specimen_id, order_provider_id,
                     storetime, priority, comments

────────────────────────────────────────────────────────────────────────────────
7. services.csv
   Purpose: Clinical service assignments — psychiatric involvement
   Stage:   1, 2 (service trajectory, psych consult detection)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   transfertime         DATETIME         Service transfer time
   prev_service         VARCHAR          Previous service
   curr_service         VARCHAR          Current service (look for "PSYCH")

   Key uses:
   • curr_service="PSYCH" → confirms psychiatric involvement
   • Service transitions (MED→PSYCH = decompensation during medical stay)

────────────────────────────────────────────────────────────────────────────────
8. transfers.csv
   Purpose: Unit transfers — acuity tracking
   Stage:   1, 2 (escalation detection, ward trajectory)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   eventtype            VARCHAR          admit / transfer / discharge
   careunit             VARCHAR          Care unit name
   intime               DATETIME         Transfer-in time
   outtime              DATETIME         Transfer-out time

   Key uses:
   • careunit containing "Psych" → psychiatric unit stay
   • Transfer to ICU = severity escalation
   • Number of transfers = instability indicator

────────────────────────────────────────────────────────────────────────────────
9. drg_severity.csv   (source: drgcodes)
   Purpose: Pre-computed severity and mortality risk scores
   Stage:   1, 2, 3 (severity baseline, outcome prediction, causal weight)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   drg_type             VARCHAR          DRG system (APR / HCFA)
   drg_code             VARCHAR          DRG code
   description          VARCHAR          DRG description
   drg_severity         INT              Severity of illness (1-4)
   drg_mortality        INT              Risk of mortality (1-4)

   Key uses:
   • drg_severity escalation across admissions = worsening trajectory
   • drg_mortality as ground-truth severity label
   • APR-DRG descriptions often mention psych conditions directly

────────────────────────────────────────────────────────────────────────────────
10. procedures_icd.csv
    Purpose: Clinical procedures performed (ECT, psych evaluations)
    Stage:   1, 2 (treatment intensity, intervention detection)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   seq_num              INT              Procedure priority
   chartdate            DATE             Procedure date
   icd_code             VARCHAR     FK → reference/d_icd_procedures
   icd_version          INT              ICD version (9 or 10)

   Key uses:
   • ECT procedure codes → treatment-resistant depression
   • Psych evaluation codes → formal psychiatric assessment
   • Join with reference/d_icd_procedures.csv for procedure names

────────────────────────────────────────────────────────────────────────────────
11. vitals_omr.csv   (source: omr)
    Purpose: Outpatient vitals and measurements (BMI, BP)
    Stage:   1, 3 (physical health comorbidity, metabolic syndrome tracking)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   chartdate            DATE             Measurement date
   result_name          VARCHAR          Metric (BMI, Blood Pressure, Weight, etc.)
   result_value         VARCHAR          Metric value

   Key uses:
   • BMI trends → metabolic syndrome from antipsychotics
   • Blood pressure → autonomic effects of psychotropics
   • Weight gain trajectory → antipsychotic side effect monitoring

────────────────────────────────────────────────────────────────────────────────
12. icu_stays.csv   (source: icustays)
    Purpose: ICU admission metadata — severity marker
    Stage:   1, 2 (ICU escalation detection, extreme outcome labeling)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   stay_id              INT         PK   Unique ICU stay ID
   first_careunit       VARCHAR          First ICU care unit
   last_careunit        VARCHAR          Last ICU care unit
   intime               DATETIME         ICU admission time
   outtime              DATETIME         ICU discharge time
   los                  FLOAT            Length of stay (days)

   Key uses:
   • Existence of ICU stay for MH patient = extreme outcome
   • los → severity duration
   • first_careunit → type of ICU (medical vs surgical vs neuro)
   • Bridge table: links hadm_id to stay_id for all ICU event tables

────────────────────────────────────────────────────────────────────────────────
13. icu_chart_events.csv   (source: chartevents, FILTERED to MH-relevant items)
    Purpose: Bedside charting — mental status, vitals, safety measures
    Stage:   1, 2, 3 (real-time clinical state, severity signals)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   stay_id              INT         FK → icu_stays
   charttime            DATETIME         Charting time
   itemid               INT         FK → reference/icu_items.csv
   value                VARCHAR          Charted value (text)
   valuenum             FLOAT            Charted value (numeric)
   valueuom             VARCHAR          Unit of measure

   Included item categories:
   • GCS (Glasgow Coma Scale) → consciousness level
   • RASS (Richmond Agitation-Sedation Scale) → agitation/sedation
   • CAM-ICU → delirium screening
   • Restraint documentation → physical restraint use
   • Suicide precautions → safety measures in place
   • Vital signs (HR, BP, RR, SpO2, Temp) → autonomic stability
   • Pain scores → distress level
   • Neurological assessments

   Columns DROPPED: caregiver_id, storetime, warning

────────────────────────────────────────────────────────────────────────────────
14. icu_input_events.csv   (source: inputevents, FILTERED to MH-relevant items)
    Purpose: IV/enteral medications given in ICU
    Stage:   1, 2, 3 (acute treatment, chemical restraint detection)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   stay_id              INT         FK → icu_stays
   starttime            DATETIME         Infusion start
   endtime              DATETIME         Infusion end
   itemid               INT         FK → reference/icu_items.csv
   amount               FLOAT            Amount administered
   amountuom            VARCHAR          Amount unit
   rate                 FLOAT            Infusion rate
   rateuom              VARCHAR          Rate unit
   ordercategorydescription VARCHAR      Order category
   statusdescription    VARCHAR          Status (Running/Stopped/etc.)

   Key uses:
   • IV Haloperidol/Lorazepam → acute agitation management
   • IV sedation (propofol, dexmedetomidine) → severe agitation/intubation
   • Naloxone → opioid overdose reversal
   • Rate escalation → worsening clinical state

   Columns DROPPED: caregiver_id, storetime, orderid, linkorderid,
   ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription,
   patientweight, totalamount, totalamountuom, isopenbag,
   continueinnextdept, originalamount, originalrate

────────────────────────────────────────────────────────────────────────────────
15. icu_output_events.csv   (source: outputevents)
    Purpose: Patient output measurements in ICU
    Stage:   1 (fluid balance, kidney function proxy)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   stay_id              INT         FK → icu_stays
   charttime            DATETIME         Charting time
   itemid               INT         FK → reference/icu_items.csv
   value                VARCHAR          Output value
   valueuom             VARCHAR          Unit of measure

   Columns DROPPED: caregiver_id, storetime

────────────────────────────────────────────────────────────────────────────────
16. icu_procedure_events.csv   (source: procedureevents)
    Purpose: ICU procedures — restraints, lines, interventions
    Stage:   1, 2, 3 (intervention detection, severity escalation)
────────────────────────────────────────────────────────────────────────────────
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   stay_id              INT         FK → icu_stays
   starttime            DATETIME         Procedure start
   endtime              DATETIME         Procedure end
   itemid               INT         FK → reference/icu_items.csv
   value                VARCHAR          Procedure value
   valueuom             VARCHAR          Unit
   statusdescription    VARCHAR          Status

   Key uses:
   • Physical restraint events → extreme agitation
   • Intubation → life-threatening event
   • Line placements → severity indicator

   Columns DROPPED: caregiver_id, storetime, location, locationcategory,
   orderid, linkorderid, ordercategoryname, ordercategorydescription,
   patientweight, isopenbag, continueinnextdept, originalamount, originalrate

────────────────────────────────────────────────────────────────────────────────
17. discharge_notes.csv   (source: Dataset2 discharge_mental_health)
    Purpose: ★ PRIMARY LLM INPUT — free-text discharge summaries
    Stage:   1, 2 (factor extraction, narrative style learning)
────────────────────────────────────────────────────────────────────────────────
   note_id              VARCHAR     PK   Unique note identifier
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   charttime            DATETIME         Note date
   text                 TEXT             Full discharge summary containing:
                                           • History of Present Illness
                                           • Past Medical/Psychiatric History
                                           • Mental Status Exam
                                           • Hospital Course
                                           • Medications at Discharge
                                           • Discharge Condition
                                           • Follow-up Instructions

   Key uses:
   • THE primary input to the fine-tuned LLM for Stage 1 extraction
   • Narrative structure templates for Stage 2 counterfactual generation
   • Contains implicit risk/protective factors in unstructured text

   Columns DROPPED: note_type, note_seq, storetime

────────────────────────────────────────────────────────────────────────────────
18. radiology_notes.csv   (source: Dataset2 radiology_mental_health)
    Purpose: Radiology reports — brain imaging findings
    Stage:   1 (organic cause detection, structural brain findings)
────────────────────────────────────────────────────────────────────────────────
   note_id              VARCHAR     PK   Unique note identifier
   subject_id           INT         FK → patients
   hadm_id              INT         FK → admissions
   charttime            DATETIME         Report date
   text                 TEXT             Radiology report (CT head, MRI brain, etc.)

   Key uses:
   • Brain imaging findings → organic vs functional psychiatric illness
   • Structural abnormalities → differential diagnosis
   • Normal imaging → supports primary psychiatric diagnosis

   Columns DROPPED: note_type, note_seq, storetime


════════════════════════════════════════════════════════════════════════════════
                         REFERENCE TABLES
════════════════════════════════════════════════════════════════════════════════

reference/mh_icd_codes.csv     — Mental health ICD codes (ICD-9: 290-319, ICD-10: F01-F99)
reference/d_icd_diagnoses.csv  — ALL ICD diagnosis codes (for comorbidity lookup)
reference/d_icd_procedures.csv — ALL ICD procedure codes (for procedure name lookup)
reference/lab_items.csv        — Lab test dictionary (MH-relevant items only)
reference/icu_items.csv        — ICU charting item dictionary (MH-relevant items only)


════════════════════════════════════════════════════════════════════════════════
                         KEY RELATIONSHIPS
════════════════════════════════════════════════════════════════════════════════

  patients.subject_id    ──1:N──►  admissions.subject_id
  admissions.hadm_id     ──1:N──►  diagnoses_icd, prescriptions,
                                   medication_admin, lab_results,
                                   services, transfers, drg_severity,
                                   procedures_icd, discharge_notes,
                                   radiology_notes, icu_stays
  icu_stays.stay_id      ──1:N──►  icu_chart_events, icu_input_events,
                                   icu_output_events, icu_procedure_events
  patients.subject_id    ──1:N──►  vitals_omr (via subject_id only)

  diagnoses_icd.icd_code ──N:1──►  reference/d_icd_diagnoses.icd_code
  procedures_icd.icd_code──N:1──►  reference/d_icd_procedures.icd_code
  lab_results.itemid     ──N:1──►  reference/lab_items.itemid
  icu_*_events.itemid    ──N:1──►  reference/icu_items.itemid


════════════════════════════════════════════════════════════════════════════════
                         TABLES DROPPED (and why)
════════════════════════════════════════════════════════════════════════════════

  emar_detail       61M rows   Detailed dose info — emar already has med + event
  poe               24M rows   Provider order entry — redundant with prescriptions
  poe_detail         5M rows   Order details — not needed
  pharmacy           8M rows   Dispensing details — redundant with prescriptions
  hcpcsevents       80K rows   Billing events — not clinical
  d_hcpcs           89K rows   HCPCS reference — not needed
  provider          42K rows   Just provider IDs (no attributes)
  caregiver         17K rows   Just caregiver IDs (no attributes)
  microbiologyevents 844K rows Culture results — not central to MH
  datetimeevents     4.9M rows ICU datetime stamps — not critical
  ingredientevents   7.1M rows IV ingredient details — too granular
  discharge_detail  126K rows  Note metadata (author) — not needed
  radiology_detail   3.5M rows Note metadata (exam codes) — not needed

================================================================================
"""


if __name__ == "__main__":
    main()