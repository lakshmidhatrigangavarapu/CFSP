"""
Phase 1: Silver Label Extraction
=================================
Uses the fine-tuned Phi-3.5-mini + LoRA adapter to extract structured
clinical factors (silver labels) from raw clinical report text.

The extraction follows the DSM-5/ICD-11 grounded schema defined in
training_data/extraction_schema.json — the same schema used during
model fine-tuning.

Output: A structured JSON matching the silver_label format in test.jsonl.
"""

import json
import logging

from .config import EXTRACTION_MAX_NEW_TOKENS, EXTRACTION_TEMPERATURE
from .model_loader import call_model
from .utils import clean_report_text, get_schema_string, safe_parse_json

log = logging.getLogger(__name__)

# Maximum clinical note characters (matches training: MAX_NOTE_CHARS = 6000)
MAX_NOTE_CHARS = 6000

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

# Finetuned model: MUST match the EXACT prompt used during training
# (from scripts/train_dgx.py SYSTEM_PROMPT)
FINETUNED_SYSTEM_PROMPT = """\
You are a clinical AI assistant specialized in mental health. 
Extract structured clinical factors from the patient note according to the schema.
Output valid JSON only."""

# Base model: more detailed instructions since it hasn't been finetuned
BASE_MODEL_SYSTEM_PROMPT = """\
You are a clinical AI assistant specialized in mental health documentation \
analysis. Extract structured clinical factors from the given clinical note \
and return them as a single valid JSON object matching the provided schema.

Rules:
- Extract ONLY information explicitly stated or clearly implied in the note.
- Ignore redacted/blank fields (e.g. ___, [REDACTED]).
- Return ONLY valid JSON. No markdown, no explanation, no extra text."""

# ============================================================================
# USER PROMPT BUILDERS
# ============================================================================


def _build_finetuned_prompt(report_text: str) -> str:
    """
    Build user prompt matching the EXACT format used during finetuning.
    (from scripts/train_dgx.py format_prompt)
    """
    cleaned = clean_report_text(report_text)
    # Truncate to match training
    if len(cleaned) > MAX_NOTE_CHARS:
        cleaned = cleaned[:MAX_NOTE_CHARS] + "... [truncated]"

    schema_str = get_schema_string()

    return (
        f"### Clinical Note:\n{cleaned}\n\n"
        f"### Extraction Schema:\n{schema_str}\n\n"
        f"### Instructions:\n"
        f"Extract all relevant clinical factors from the note. "
        f"Return a JSON object matching the schema."
    )


def _build_base_prompt(report_text: str) -> str:
    """Build user prompt for the base model (simplified schema to save tokens)."""
    cleaned = clean_report_text(report_text)
    if len(cleaned) > MAX_NOTE_CHARS:
        cleaned = cleaned[:MAX_NOTE_CHARS] + "... [truncated]"

    # Condensed schema — field names and types only, saves ~1000 tokens
    # vs the full nested schema. Enough for the base model to understand structure.
    condensed_schema = """\
{
  "primary_mh_diagnosis": {"icd_code": "str", "title": "str", "dsm5_category": "str"},
  "comorbid_mh_diagnoses": [{"icd_code": "str", "title": "str"}],
  "medical_comorbidities": ["str"],
  "severity_level": "mild|moderate|severe|critical",
  "suicide_risk_indicators": {
    "ideation_mentioned": true/false,
    "attempt_history": true/false,
    "self_harm": true/false,
    "precautions_ordered": true/false,
    "risk_level": "none|low|moderate|high"
  },
  "substance_use": {
    "alcohol": "none|active|history",
    "drugs": "none|active|history",
    "tobacco": "none|active|history",
    "positive_tox_screen": true/false,
    "substances_detected": ["str"]
  },
  "medication_profile": {
    "psychotropic_medications": [{"drug": "str", "class": "str"}],
    "polypharmacy": true/false,
    "medication_count": number
  },
  "social_factors": {
    "marital_status": "str",
    "insurance_type": "str",
    "social_isolation_risk": true/false
  },
  "admission_context": {
    "admission_type": "str",
    "is_emergency": true/false,
    "los_days": number,
    "discharge_disposition": "str"
  },
  "psychiatric_service_involvement": true/false,
  "lab_abnormalities": [{"test": "str", "finding": "str"}]
}"""

    return (
        "### Clinical Note ###\n"
        f"{cleaned}\n\n"
        "### Target Schema ###\n"
        f"{condensed_schema}\n\n"
        "### Instructions ###\n"
        "Read the clinical note carefully. Extract ALL fields from the schema.\n"
        "Return ONLY a valid JSON object. No markdown, no extra text.\n"
        "For missing information use null for strings, false for booleans, [] for arrays.\n"
        "If the primary condition is medical (not psychiatric), still extract what you can "
        "and set severity_level to the MH severity (or 'mild' if no MH issues found).\n"
    )


# ============================================================================
# EXTRACTION ENTRYPOINT
# ============================================================================


def extract_silver_labels(
    report_text: str,
    model,
    tokenizer,
    device: str,
    use_finetuned: bool = True,
) -> dict:
    """
    Extract structured silver labels from a clinical report.

    Args:
        report_text: Raw clinical note text (may contain redacted fields).
        model: Loaded model (with or without LoRA adapters).
        tokenizer: Loaded tokenizer.
        device: CUDA device string.
        use_finetuned: If True, use the raw prompt format matching the
                       finetuning template. If False, use chat_template
                       with a more detailed system prompt for the base model.

    Returns:
        Dictionary of extracted clinical factors matching the schema.
    """
    mode_str = "finetuned" if use_finetuned else "base-model"
    log.info(f"Phase 1: Extracting silver labels ({mode_str} mode) ...")

    if use_finetuned:
        system_prompt = FINETUNED_SYSTEM_PROMPT
        user_prompt = _build_finetuned_prompt(report_text)
    else:
        system_prompt = BASE_MODEL_SYSTEM_PROMPT
        user_prompt = _build_base_prompt(report_text)

    raw_output = call_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=EXTRACTION_MAX_NEW_TOKENS,
        temperature=EXTRACTION_TEMPERATURE,
        use_raw_prompt=use_finetuned,  # raw tokens for finetuned, chat_template for base
    )

    log.info(f"Phase 1: Raw extraction output ({len(raw_output)} chars)")
    log.debug(f"Phase 1 raw: {raw_output[:500]}")

    silver_labels = safe_parse_json(raw_output)

    # safe_parse_json can return a list or dict; we need a dict
    if isinstance(silver_labels, list):
        # Model returned an array — wrap it or treat as failed
        log.warning("Phase 1: Model returned a JSON array instead of object. Wrapping.")
        silver_labels = {"_raw_array": silver_labels, "_parse_failed": True}

    if silver_labels.get("_parse_failed"):
        log.error("Phase 1: Failed to parse extraction output as JSON.")
        log.error(f"Raw preview: {raw_output[:500]}")
    else:
        dx = silver_labels.get("primary_mh_diagnosis", {})
        log.info(
            f"Phase 1 DONE - Primary Dx: {dx.get('title', '?')}, "
            f"Severity: {silver_labels.get('severity_level', '?')}"
        )

    return silver_labels
