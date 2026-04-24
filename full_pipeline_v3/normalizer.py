"""
Phase 1.5: Post-Extraction Normalizer (NEW in V2)
===================================================
Validates and corrects Phase 1 silver label extraction using:

1. ICD code validation — maps codes to correct disease categories
2. Discharge diagnosis reconciliation — cross-checks extracted Dx
   against the Discharge Diagnosis section of the note
3. Disease category classification — psychiatric vs neurodegenerative
   vs medical
4. Keyword-based disease detection — catches redacted disease names
   (e.g., "[REDACTED] Disease" → Huntington's from context clues)

This runs BETWEEN Phase 1 and Phase 2, fixing errors before they
propagate downstream.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .config import ICD_LOOKUP_PATH

log = logging.getLogger(__name__)


# ============================================================================
# ICD LOOKUP TABLE
# ============================================================================

_icd_lookup = None


def _load_icd_lookup() -> dict:
    """Load ICD lookup table (cached)."""
    global _icd_lookup
    if _icd_lookup is None:
        path = Path(ICD_LOOKUP_PATH)
        if path.exists():
            with open(path) as f:
                _icd_lookup = json.load(f)
            log.info(f"Loaded ICD lookup table from {path}")
        else:
            log.warning(f"ICD lookup table not found at {path}")
            _icd_lookup = {}
    return _icd_lookup


def _find_icd_category(icd_code: str) -> Optional[dict]:
    """
    Look up an ICD code in the lookup table.
    Returns the category dict or None.
    """
    lookup = _load_icd_lookup()
    if not icd_code:
        return None

    code_upper = icd_code.upper().strip()

    for category_name, category_data in lookup.items():
        if category_name.startswith("_"):
            continue
        codes = category_data.get("codes", [])
        # Check exact match first, then prefix match
        for c in codes:
            if code_upper == c.upper() or code_upper.startswith(c.upper()):
                return {
                    "category_name": category_name,
                    "disease_category": category_data.get("disease_category"),
                    "correct_dsm5": category_data.get("correct_dsm5"),
                    "note": category_data.get("note", ""),
                    "specific": category_data.get("specific", {}).get(code_upper, {}),
                }
    return None


def _detect_disease_from_keywords(note_text: str) -> Optional[dict]:
    """
    Detect specific diseases from clinical note text using keyword patterns.
    Handles redacted disease names by looking for contextual clues.
    """
    lookup = _load_icd_lookup()
    keyword_map = lookup.get("_keyword_to_icd", {})

    note_lower = note_text.lower()

    # Pattern 1: Direct keyword match
    for keyword, icd_code in keyword_map.items():
        if keyword.lower() in note_lower:
            category = _find_icd_category(icd_code)
            if category:
                category["detected_keyword"] = keyword
                category["detected_icd"] = icd_code
                return category

    # Pattern 2: Contextual detection for redacted Huntington's Disease
    # Look for: "[REDACTED] Disease" + "chorea" + psychiatric admission
    huntingtons_clues = 0
    if re.search(r'\[REDACTED\]\s*disease', note_lower):
        huntingtons_clues += 1
    if 'chorea' in note_lower:
        huntingtons_clues += 1
    if re.search(r'choreiform\s+movements?', note_lower):
        huntingtons_clues += 1
    if re.search(r'mother.*(?:disease|died|diagnosed)', note_lower):
        huntingtons_clues += 1  # hereditary pattern
    if re.search(r'movement\s+disorder', note_lower):
        huntingtons_clues += 1
    if re.search(r'dysarthri[ca]', note_lower):
        huntingtons_clues += 1

    if huntingtons_clues >= 3:
        log.info(f"Keyword detection: Huntington's Disease suspected ({huntingtons_clues} clues)")
        return {
            "category_name": "movement_disorders",
            "disease_category": "neurodegenerative",
            "correct_dsm5": "neurocognitive_disorders",
            "detected_keyword": "huntingtons_contextual",
            "detected_icd": "G10",
            "note": "Detected from contextual clues: chorea + hereditary + [REDACTED] Disease",
        }

    # Pattern 3: Parkinson's contextual detection
    parkinsons_clues = 0
    if re.search(r'tremor', note_lower):
        parkinsons_clues += 1
    if re.search(r'rigidity', note_lower):
        parkinsons_clues += 1
    if re.search(r'bradykinesia', note_lower):
        parkinsons_clues += 1
    if re.search(r'levodopa|carbidopa|sinemet', note_lower):
        parkinsons_clues += 1

    if parkinsons_clues >= 3:
        log.info(f"Keyword detection: Parkinson's Disease suspected ({parkinsons_clues} clues)")
        return {
            "category_name": "movement_disorders",
            "disease_category": "neurodegenerative",
            "correct_dsm5": "neurocognitive_disorders",
            "detected_keyword": "parkinsons_contextual",
            "detected_icd": "G20",
            "note": "Detected from contextual clues: tremor + rigidity + bradykinesia",
        }

    return None


def _extract_discharge_section(note_text: str) -> str:
    """Extract the Discharge Diagnosis section from a clinical note."""
    patterns = [
        r'Discharge\s+Diagnos[ie]s?[:\s]*\n(.*?)(?:\n\s*\n|\nDischarge\s+Condition)',
        r'Discharge\s+Diagnos[ie]s?[:\s]*\n(.*?)(?:\n\s*\n|\n[A-Z])',
        r'Discharge\s+Dx[:\s]*\n(.*?)(?:\n\s*\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


# ============================================================================
# MAIN NORMALIZER
# ============================================================================


def normalize_silver_labels(
    silver_labels: dict,
    note_text: str,
) -> dict:
    """
    Post-extraction normalization and validation of silver labels.

    Checks:
    1. ICD code → correct disease category mapping
    2. Discharge diagnosis reconciliation
    3. Disease keyword detection from note text
    4. dsm5_category correction if wrong
    5. Add disease_category, progression if missing

    Args:
        silver_labels: Extracted silver labels from Phase 1.
        note_text: Original clinical note text (cleaned).

    Returns:
        Corrected silver_labels dict with _normalization_log listing changes.
    """
    if not silver_labels or silver_labels.get("_parse_failed"):
        return silver_labels

    changes = []
    dx = silver_labels.get("primary_mh_diagnosis") or {}
    if not isinstance(dx, dict):
        dx = {}

    extracted_icd = dx.get("icd_code", "")
    extracted_dsm5 = dx.get("dsm5_category", "")
    extracted_title = dx.get("title", "")

    # ----------------------------------------------------------------
    # CHECK 1: ICD code validation
    # ----------------------------------------------------------------
    icd_result = _find_icd_category(extracted_icd) if extracted_icd else None

    if icd_result:
        correct_dsm5 = icd_result.get("correct_dsm5", "")
        correct_disease_cat = icd_result.get("disease_category", "")

        if correct_dsm5 and extracted_dsm5 != correct_dsm5:
            changes.append(
                f"ICD validation: dsm5_category '{extracted_dsm5}' → '{correct_dsm5}' "
                f"(ICD {extracted_icd} maps to {icd_result['category_name']})"
            )
            dx["dsm5_category"] = correct_dsm5

        if correct_disease_cat:
            old_cat = silver_labels.get("disease_category", "unknown")
            if old_cat != correct_disease_cat:
                changes.append(
                    f"ICD validation: disease_category '{old_cat}' → '{correct_disease_cat}'"
                )
            silver_labels["disease_category"] = correct_disease_cat

        # Update progression from specific code data
        specific = icd_result.get("specific", {})
        if specific and specific.get("progression"):
            old_prog = silver_labels.get("progression", "unknown")
            silver_labels["progression"] = specific["progression"]
            if old_prog != specific["progression"]:
                changes.append(
                    f"ICD validation: progression '{old_prog}' → '{specific['progression']}'"
                )

    # ----------------------------------------------------------------
    # CHECK 2: Keyword-based disease detection from note text
    # ----------------------------------------------------------------
    keyword_result = _detect_disease_from_keywords(note_text)

    if keyword_result:
        detected_icd = keyword_result.get("detected_icd", "")
        detected_dsm5 = keyword_result.get("correct_dsm5", "")
        detected_cat = keyword_result.get("disease_category", "")

        # If the model extracted a symptom (e.g., "Chorea") but we detected
        # the actual disease (e.g., Huntington's), update
        symptom_codes = {"G25.5", "G25.1", "G25.4"}
        if extracted_icd in symptom_codes and detected_icd not in symptom_codes:
            changes.append(
                f"Keyword detection: Replaced symptom ICD '{extracted_icd}' "
                f"with disease ICD '{detected_icd}' ({keyword_result.get('detected_keyword')})"
            )
            dx["icd_code"] = detected_icd

        # Fix dsm5_category if keyword detection found a better one
        if detected_dsm5 and dx.get("dsm5_category") != detected_dsm5:
            # Only override if the keyword detection is more specific
            if dx.get("dsm5_category") in ("neurodevelopmental_disorders", "other_mental_health", ""):
                changes.append(
                    f"Keyword detection: dsm5_category '{dx.get('dsm5_category')}' → '{detected_dsm5}'"
                )
                dx["dsm5_category"] = detected_dsm5

        if detected_cat:
            silver_labels["disease_category"] = detected_cat

    # ----------------------------------------------------------------
    # CHECK 3: Discharge diagnosis reconciliation
    # ----------------------------------------------------------------
    discharge_section = _extract_discharge_section(note_text)
    if discharge_section:
        discharge_lines = [
            line.strip().lstrip("- •").strip()
            for line in discharge_section.split("\n")
            if line.strip() and line.strip() not in ("[REDACTED]", "")
        ]

        # Store discharge diagnoses for downstream use
        if discharge_lines and not silver_labels.get("discharge_diagnoses"):
            silver_labels["discharge_diagnoses"] = discharge_lines
            changes.append(f"Added discharge_diagnoses: {discharge_lines}")

        # Check for "mood disorder" in discharge dx
        discharge_lower = discharge_section.lower()
        has_mood_disorder = any(
            phrase in discharge_lower
            for phrase in [
                "mood disorder",
                "depressive disorder",
                "bipolar",
                "mood dysregulation",
            ]
        )

        has_secondary = any(
            phrase in discharge_lower
            for phrase in [
                "secondary to",
                "due to",
                "general medical condition",
            ]
        )

        if has_mood_disorder:
            # Check if the current primary Dx captures the mood disorder
            current_title_lower = extracted_title.lower()
            if "mood" not in current_title_lower and "depressive" not in current_title_lower:
                # The model missed the mood disorder — it's probably extracting
                # the underlying medical condition instead
                if has_secondary:
                    changes.append(
                        f"Discharge Dx reconciliation: Primary Dx '{extracted_title}' "
                        f"doesn't capture mood disorder. Discharge lists mood disorder "
                        f"secondary to medical condition."
                    )
                    # Add mood disorder as primary, move current to medical comorbidity
                    if dx.get("title"):
                        med_comorbs = silver_labels.get("medical_comorbidities") or []
                        if dx["title"] not in med_comorbs:
                            med_comorbs.append(dx["title"])
                            silver_labels["medical_comorbidities"] = med_comorbs

                    dx["title"] = "Mood Disorder secondary to General Medical Condition"
                    dx["icd_code"] = "F06.3"
                    dx["dsm5_category"] = "mood_disorder_due_to_medical_condition"
                    silver_labels["disease_category"] = "psychiatric_secondary_to_medical"
                    changes.append(
                        "Updated primary Dx to 'Mood Disorder secondary to GMC' (F06.3)"
                    )

    # ----------------------------------------------------------------
    # CHECK 4: Ensure disease_category is set
    # ----------------------------------------------------------------
    if not silver_labels.get("disease_category") or silver_labels["disease_category"] == "unknown":
        dsm5 = dx.get("dsm5_category", "")
        if dsm5 in ("depressive_disorders", "bipolar_disorders", "anxiety_disorders",
                     "trauma_stress_disorders", "schizophrenia_psychotic",
                     "substance_use_disorders", "personality_disorders",
                     "neurodevelopmental_disorders"):
            silver_labels["disease_category"] = "psychiatric_primary"
        elif dsm5 == "neurocognitive_disorders":
            silver_labels["disease_category"] = "neurodegenerative"
        elif dsm5 == "mood_disorder_due_to_medical_condition":
            silver_labels["disease_category"] = "psychiatric_secondary_to_medical"
        else:
            silver_labels["disease_category"] = "unknown"
        changes.append(f"Inferred disease_category: {silver_labels['disease_category']}")

    # ----------------------------------------------------------------
    # CHECK 5: Neurodevelopmental vs Neurodegenerative guard
    # ----------------------------------------------------------------
    if dx.get("dsm5_category") == "neurodevelopmental_disorders":
        # Neurodevelopmental should be: intellectual disability, autism, ADHD, etc.
        # NOT: Huntington's, Parkinson's, Alzheimer's
        neurodegen_keywords = [
            "huntington", "parkinson", "alzheimer", "dementia",
            "chorea", "neurodegen", "progressive", "lewy body",
        ]
        title_lower = dx.get("title", "").lower()
        if any(kw in title_lower for kw in neurodegen_keywords):
            changes.append(
                f"Guard: '{dx['title']}' misclassified as neurodevelopmental. "
                f"Correcting to neurocognitive_disorders."
            )
            dx["dsm5_category"] = "neurocognitive_disorders"
            silver_labels["disease_category"] = "neurodegenerative"

    # ----------------------------------------------------------------
    # WRITE BACK
    # ----------------------------------------------------------------
    silver_labels["primary_mh_diagnosis"] = dx

    if changes:
        silver_labels["_normalization_log"] = changes
        log.info(f"Post-extraction normalizer made {len(changes)} corrections:")
        for c in changes:
            log.info(f"  → {c}")
    else:
        log.info("Post-extraction normalizer: no corrections needed.")

    return silver_labels
