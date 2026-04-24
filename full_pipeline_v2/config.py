"""
Centralized configuration for the full inference pipeline V2.

V2 changes:
  - Added validation/normalization settings
  - Added critical signal detection config
  - Added branch gating thresholds
  - Extended schema path
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PIPELINE_DIR = Path(__file__).parent
PACKAGE_DIR = PIPELINE_DIR.parent

# Finetuned model for silver-label extraction (Phase 1)
BASE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Checkpoint paths
CHECKPOINT_PATH = str(PACKAGE_DIR / "output_v2" / "best_model")
if not Path(CHECKPOINT_PATH).exists():
    CHECKPOINT_PATH = str(PACKAGE_DIR / "output" / "checkpoint-1500")

# Extraction schema — use the SAME V1 schema the model was finetuned on.
# V2 enrichment (disease_category, progression, etc.) is done post-extraction
# by the normalizer in Phase 1.5.
SCHEMA_PATH = str(PACKAGE_DIR / "training_data" / "extraction_schema.json")

# ICD code lookup table
ICD_LOOKUP_PATH = str(PIPELINE_DIR / "icd_lookup.json")

# Default output
OUTPUT_DIR = str(PIPELINE_DIR / "output")

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

# Phase 1: Silver label extraction
EXTRACTION_MAX_NEW_TOKENS = 1536
EXTRACTION_TEMPERATURE = 0.0

# Phase 1.5: Discharge diagnosis re-extraction
DISCHARGE_DX_MAX_NEW_TOKENS = 512
DISCHARGE_DX_TEMPERATURE = 0.0

# Phase 2: Tree-of-Thoughts scenario generation
TOT_MAX_NEW_TOKENS_STEP1 = 512
TOT_MAX_NEW_TOKENS_STEP2 = 512
TOT_MAX_NEW_TOKENS_STEP3 = 1280
TOT_TEMPERATURE = 0.3

# Phase 3: Patient report
REPORT_MAX_NEW_TOKENS = 2048
REPORT_TEMPERATURE = 0.2

# ============================================================================
# V2: VALIDATION & GATING
# ============================================================================

# Branch gating: minimum evidence threshold to run a branch
# If a branch's pre-flight relevance score is below this, skip it
BRANCH_GATE_MIN_EVIDENCE_FACTORS = 1  # need at least 1 supporting factor

# Critical signal keywords — trigger HIGH risk override
CRITICAL_SIGNAL_PATTERNS = {
    "suicide": [
        r"suicid\w*",
        r"jump\s+(?:out|off|from)",
        r"hang\s+(?:my|him|her)self",
        r"kill\s+(?:my|him|her)self",
        r"want\s+to\s+die",
        r"end\s+(?:my|his|her)\s+life",
        r"overdose\s+(?:on|with)",
        r"cut\s+(?:my|him|her)self",
        r"self[- ]?harm",
        r"wrist\s+cut",
    ],
    "violence": [
        r"assault\w*",
        r"physically\s+(?:attack|hit|strik|beat|hurt)",
        r"threaten\w*\s+(?:to\s+)?(?:kill|hurt|harm|stab|shoot)",
        r"homicid\w*",
        r"violen\w+",
        r"aggress\w+",
        r"restrain\w*",
        r"weapon\w*",
        r"strangle",
        r"choke",
    ],
    "self_harm": [
        r"self[- ]?injur\w*",
        r"self[- ]?harm",
        r"cut(?:ting)?\s+(?:my|him|her)self",
        r"burn(?:ing)?\s+(?:my|him|her)self",
        r"head[- ]?bang",
    ],
}

# ============================================================================
# ToT BRANCH DEFINITIONS (V2 — with gating conditions)
# ============================================================================

NUM_BRANCHES = 3

BRANCHES = [
    {
        "id": "A",
        "type": "clinical_deterioration",
        "label": "Psychiatric / Clinical Deterioration Pathway",
        "gate_condition": "always",
        "focus": (
            "Focus on how the patient's PRIMARY psychiatric or mental health "
            "condition could worsen: medication non-adherence, treatment "
            "resistance, relapse, acute decompensation, re-hospitalisation, "
            "emergence of psychosis, or escalating self-harm. "
            "ONLY consider deterioration pathways that are DIRECTLY relevant "
            "to the patient's mental health. If the patient's primary problem "
            "is a non-psychiatric medical condition (e.g., lung disease, "
            "cardiac issue), state clearly that this pathway has LOW mental "
            "health relevance and explain what limited psychiatric impact, "
            "if any, exists (e.g., adjustment disorder, medication side effects)."
        ),
    },
    {
        "id": "B",
        "type": "substance_escalation",
        "label": "Substance Use Escalation Pathway",
        "gate_condition": "evidence_required",
        "gate_fields": [
            "substance_use.alcohol",
            "substance_use.drugs",
            "substance_use.positive_tox_screen",
        ],
        "gate_values_skip": ["none", False, None],
        "focus": (
            "Focus on how the patient could develop or worsen substance use "
            "(alcohol, opioids, stimulants, etc.) as a coping mechanism or "
            "due to untreated pain/distress — leading to comorbid SUD, "
            "impaired judgment, overdose risk, or worsened psychiatric outcomes. "
            "If the patient has NO substance use history and NO clear risk "
            "factors for substance abuse, state that this pathway has LOW "
            "relevance and briefly explain why. Do NOT fabricate a substance "
            "abuse trajectory when there is no clinical basis for it."
        ),
    },
    {
        "id": "C",
        "type": "social_environmental_collapse",
        "label": "Social / Environmental Collapse Pathway",
        "gate_condition": "always",
        "focus": (
            "Focus on how the patient's social and environmental safety net "
            "could erode: housing instability, relationship breakdown, "
            "unemployment, social isolation, loss of family support — "
            "removing protective factors and accelerating mental health crisis "
            "without a recovery anchor. "
            "If the patient has STRONG social support and stable environment, "
            "acknowledge that and describe only the most plausible (even if "
            "unlikely) erosion scenario. Always tie back to mental health impact."
        ),
    },
]
