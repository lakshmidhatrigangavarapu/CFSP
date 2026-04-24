"""
Centralized configuration for the full inference pipeline.
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

# Checkpoint paths - switch between V1 and V2 trained models
# V1 (original training): output/checkpoint-1500
# V2 (aligned prompts):   output_v2/best_model (after running train_dgx_v2.py)
CHECKPOINT_PATH = str(PACKAGE_DIR / "output_v2" / "best_model")
# Fallback to V1 if V2 doesn't exist yet
if not Path(CHECKPOINT_PATH).exists():
    CHECKPOINT_PATH = str(PACKAGE_DIR / "output" / "checkpoint-1500")

# Extraction schema
SCHEMA_PATH = str(PACKAGE_DIR / "training_data" / "extraction_schema.json")

# Default output
OUTPUT_DIR = str(PIPELINE_DIR / "output")

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

# Phase 1: Silver label extraction
EXTRACTION_MAX_NEW_TOKENS = 1536
EXTRACTION_TEMPERATURE = 0.0  # deterministic for structured extraction

# Phase 2: Tree-of-Thoughts scenario generation
TOT_MAX_NEW_TOKENS_STEP1 = 512   # trigger identification
TOT_MAX_NEW_TOKENS_STEP2 = 512   # causal chain reasoning
TOT_MAX_NEW_TOKENS_STEP3 = 1280  # narrative generation
TOT_TEMPERATURE = 0.3            # slight creativity for scenarios

# Phase 3: Patient report
REPORT_MAX_NEW_TOKENS = 2048
REPORT_TEMPERATURE = 0.2

# ============================================================================
# ToT BRANCH DEFINITIONS
# ============================================================================

NUM_BRANCHES = 3

BRANCHES = [
    {
        "id": "A",
        "type": "clinical_deterioration",
        "label": "Psychiatric / Clinical Deterioration Pathway",
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
