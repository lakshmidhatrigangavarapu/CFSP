"""
Phase 2: Tree-of-Thoughts (ToT) Scenario Generation
=====================================================
Generates 2-3 distinct negative patient trajectory scenarios using a
structured Tree-of-Thoughts approach.

Each branch explores ONE specific escalation pathway independently:
  Branch A — Psychiatric / Clinical Deterioration
  Branch B — Substance Use Escalation
  Branch C — Social / Environmental Collapse

For each branch, the pipeline runs three deliberate reasoning steps:
  Step 1 (Identify)  — Which factors in the patient's current state can
                        ignite this pathway?
  Step 2 (Reason)    — Trace the causal chain: trigger → escalation → crisis.
  Step 3 (Narrate)   — Generate a concrete, clinically plausible scenario
                        narrative.

IMPORTANT — Mental Health Relevance Filter:
  If the patient's condition is primarily non-psychiatric (e.g., lung
  disease, cardiac surgery), the model is instructed to:
  - Acknowledge the limited mental health impact
  - NOT fabricate unrelated psychiatric deterioration
  - Focus ONLY on genuine mental health consequences (e.g., adjustment
    disorders, medication side effects, iatrogenic depression)
"""

import json
import logging
from typing import Optional

from .config import (
    BRANCHES,
    NUM_BRANCHES,
    TOT_MAX_NEW_TOKENS_STEP1,
    TOT_MAX_NEW_TOKENS_STEP2,
    TOT_MAX_NEW_TOKENS_STEP3,
    TOT_TEMPERATURE,
)
from .model_loader import call_model
from .utils import compact_factors, safe_parse_json, parse_marked_sections, parse_bullet_list

log = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT — Scenario Generation
# ============================================================================

SCENARIO_SYSTEM_PROMPT = """\
You are a senior clinical psychiatrist and patient safety specialist with \
expertise in mental health risk assessment and counterfactual reasoning.

YOUR ROLE: Reason carefully about possible NEGATIVE future trajectories for \
a patient based on their CURRENT clinical state. You are NOT predicting \
outcomes — you are identifying plausible extreme negative scenarios so that \
clinicians can PREPARE and PREVENT them.

CRITICAL RULES:
1. GROUND ALL REASONING in the patient's actual clinical data. Never \
fabricate diagnoses, risk factors, or history not supported by the patient's \
current state.

2. MENTAL HEALTH RELEVANCE: Every scenario MUST be tied to the patient's \
mental health. If the patient's primary condition is a NON-psychiatric \
medical issue (e.g., lung disease, cardiac problem, surgical recovery), \
you MUST:
   a) Acknowledge that the primary medical condition does NOT directly \
cause mental health deterioration.
   b) Identify ONLY genuine, evidence-based psychiatric consequences: \
adjustment disorder, post-surgical depression, medication-induced mood \
changes, grief/loss reactions, etc.
   c) Rate the scenario as LOW plausibility if there is minimal mental \
health basis.
   d) NEVER invent a psychotic episode, substance abuse, or suicidal \
ideation for a patient with no psychiatric history just to fill a scenario.

3. REDACTED INFORMATION: Fields marked [REDACTED] are de-identified \
personal information (names, dates, locations). Ignore these completely — \
they are not clinically relevant.

4. CLINICAL ACCURACY: Use proper psychiatric terminology. Reference DSM-5 \
categories and ICD codes where appropriate. Cite specific patient data \
points to support each claim.

5. OUTPUT: Respond in the format requested by each step. For structured \
data, use JSON. For narratives, use plain text with clear section headers."""

# ============================================================================
# STEP 1 — Trigger Identification
# ============================================================================


def _step1_prompt(branch: dict, patient_summary: str, note_excerpt: str) -> str:
    return (
        f"### PATHWAY: {branch['label']} ###\n"
        f"Focus: {branch['focus']}\n\n"
        f"### PATIENT STATE (Extracted Clinical Factors) ###\n"
        f"{patient_summary}\n\n"
        f"### CLINICAL NOTE (Excerpt) ###\n"
        f"{note_excerpt}\n\n"
        "### TASK ###\n"
        "Identify 3-5 factors ALREADY PRESENT in this patient's data that "
        f"could trigger the '{branch['type']}' pathway.\n\n"
        "For each factor:\n"
        "- State the specific clinical data point from the patient\n"
        "- Explain WHY it is a risk point for this pathway\n"
        "- Rate its relevance to MENTAL HEALTH specifically (high/medium/low)\n\n"
        "If this pathway has LOW overall relevance to this patient's mental "
        "health (e.g., patient has no psychiatric history, primary issue is "
        "medical/surgical), state that clearly and list only the minimal "
        "factors that could apply.\n\n"
        "Reply ONLY with a JSON array:\n"
        '[{"factor": "...", "evidence": "...", "reason": "...", '
        '"mh_relevance": "high|medium|low"}]'
    )


# ============================================================================
# STEP 2 — Causal Chain Reasoning
# ============================================================================


def _step2_prompt(branch: dict, triggers_json: str, patient_summary: str) -> str:
    return (
        f"### PATHWAY: {branch['label']} ###\n\n"
        f"### TRIGGERING FACTORS IDENTIFIED ###\n"
        f"{triggers_json}\n\n"
        f"### PATIENT STATE ###\n"
        f"{patient_summary}\n\n"
        "### TASK ###\n"
        "Trace the causal chain from these triggers to a clinical crisis "
        "endpoint. Provide 4-6 sequential escalation steps.\n\n"
        "Each step must:\n"
        "- Describe a specific clinical event\n"
        "- Explain the biological/psychological/social mechanism\n"
        "- Connect clearly to the next step\n"
        "- Stay grounded in evidence from this patient's data\n\n"
        "If the triggers have LOW mental health relevance, the chain should "
        "reflect that — show a MILD trajectory (e.g., adjustment difficulty, "
        "temporary mood change) rather than fabricating a severe psychiatric "
        "crisis.\n\n"
        "Reply ONLY with a JSON array:\n"
        '[{"step": 1, "event": "...", "mechanism": "...", '
        '"connects_to": "..."}]'
    )


# ============================================================================
# STEP 3 — Scenario Narrative
# ============================================================================


def _step3_prompt(
    branch: dict,
    triggers_json: str,
    causal_chain_json: str,
    patient_summary: str,
    note_excerpt: str,
) -> str:
    return (
        f"### PATHWAY: {branch['label']} ###\n\n"
        f"### TRIGGERING FACTORS ###\n{triggers_json}\n\n"
        f"### CAUSAL CHAIN ###\n{causal_chain_json}\n\n"
        f"### PATIENT STATE ###\n{patient_summary}\n\n"
        "### TASK ###\n"
        "Generate a scenario analysis for this negative trajectory.\n"
        "If the patient's primary condition is NON-psychiatric and the "
        "triggers have LOW mental health relevance, say so honestly — "
        "do NOT invent dramatic psychiatric crises.\n\n"
        "Write your response using EXACTLY these section headers:\n\n"
        "NARRATIVE:\n"
        "(2-3 paragraphs describing the scenario in clinical language)\n\n"
        "WARNING SIGNS:\n"
        "- sign 1\n"
        "- sign 2\n"
        "(list 3-5 warning signs clinicians should watch for)\n\n"
        "CRISIS ENDPOINT:\n"
        "(worst realistic outcome in 1-2 sentences)\n\n"
        "PREPAREDNESS ACTIONS:\n"
        "- action 1\n"
        "- action 2\n"
        "(list 3-5 concrete actions clinicians can take NOW)\n\n"
        "PLAUSIBILITY: high or medium or low\n\n"
        "PLAUSIBILITY RATIONALE:\n"
        "(brief explanation of why this rating)\n\n"
        "MH RELEVANCE: high or medium or low\n\n"
        "MH RELEVANCE NOTE:\n"
        "(brief explanation of mental health relevance)"
    )


# ============================================================================
# OVERALL RELEVANCE ASSESSMENT
# ============================================================================


def _relevance_assessment_prompt(patient_summary: str, note_excerpt: str) -> str:
    """Pre-flight check: assess how relevant mental health scenarios are."""
    return (
        "### PATIENT STATE ###\n"
        f"{patient_summary}\n\n"
        "### CLINICAL NOTE (Excerpt) ###\n"
        f"{note_excerpt}\n\n"
        "### TASK ###\n"
        "Before generating counterfactual mental health scenarios, assess:\n"
        "1. Is this patient's PRIMARY condition a mental health condition?\n"
        "2. Does this patient have significant psychiatric history?\n"
        "3. How relevant are mental health scenarios for this patient?\n\n"
        "Reply ONLY with a JSON object:\n"
        '{"primary_is_psychiatric": true/false, '
        '"has_psych_history": true/false, '
        '"mh_scenario_relevance": "high|medium|low", '
        '"explanation": "brief explanation...", '
        '"recommended_focus": "what scenarios should focus on..."}'
    )


# ============================================================================
# SINGLE BRANCH EXECUTION
# ============================================================================


def _run_branch(
    branch: dict,
    patient_summary: str,
    note_excerpt: str,
    model,
    tokenizer,
    device: str,
) -> dict:
    """Execute all 3 ToT steps for a single branch."""

    # --- Step 1: Identify triggers ---
    log.info(f"  Branch {branch['id']} ({branch['type']}) — Step 1: Trigger identification")
    raw_step1 = call_model(
        model, tokenizer, device,
        system_prompt=SCENARIO_SYSTEM_PROMPT,
        user_prompt=_step1_prompt(branch, patient_summary, note_excerpt),
        max_new_tokens=TOT_MAX_NEW_TOKENS_STEP1,
        temperature=TOT_TEMPERATURE,
    )
    log.debug(f"  Branch {branch['id']} Step 1 raw: {raw_step1[:200]}")
    triggers = safe_parse_json(raw_step1)

    # Prepare clean trigger text for next step
    if isinstance(triggers, (list, dict)) and not (isinstance(triggers, dict) and triggers.get("_parse_failed")):
        triggers_json = json.dumps(triggers, indent=2)
    else:
        triggers_json = raw_step1[:500]

    # --- Step 2: Causal chain ---
    log.info(f"  Branch {branch['id']} — Step 2: Causal chain reasoning")
    raw_step2 = call_model(
        model, tokenizer, device,
        system_prompt=SCENARIO_SYSTEM_PROMPT,
        user_prompt=_step2_prompt(branch, triggers_json, patient_summary),
        max_new_tokens=TOT_MAX_NEW_TOKENS_STEP2,
        temperature=TOT_TEMPERATURE,
    )
    log.debug(f"  Branch {branch['id']} Step 2 raw: {raw_step2[:200]}")
    causal_chain = safe_parse_json(raw_step2)

    if isinstance(causal_chain, (list, dict)) and not (isinstance(causal_chain, dict) and causal_chain.get("_parse_failed")):
        chain_json = json.dumps(causal_chain, indent=2)
    else:
        chain_json = raw_step2[:500]

    # --- Step 3: Narrative generation (plain text output) ---
    log.info(f"  Branch {branch['id']} — Step 3: Scenario narrative generation")
    raw_step3 = call_model(
        model, tokenizer, device,
        system_prompt=SCENARIO_SYSTEM_PROMPT,
        user_prompt=_step3_prompt(
            branch, triggers_json, chain_json, patient_summary, note_excerpt
        ),
        max_new_tokens=TOT_MAX_NEW_TOKENS_STEP3,
        temperature=TOT_TEMPERATURE,
    )
    log.debug(f"  Branch {branch['id']} Step 3 raw: {raw_step3[:200]}")

    # Parse plain-text sections into a structured dict
    _STEP3_MARKERS = [
        "NARRATIVE", "WARNING SIGNS", "CRISIS ENDPOINT",
        "PREPAREDNESS ACTIONS", "PLAUSIBILITY RATIONALE",
        "PLAUSIBILITY", "MH RELEVANCE NOTE", "MH RELEVANCE",
    ]
    sections = parse_marked_sections(raw_step3, _STEP3_MARKERS)

    scenario = {
        "narrative": sections.get("narrative", raw_step3[:800]),
        "warning_signs": parse_bullet_list(sections.get("warning_signs", "")),
        "crisis_endpoint": sections.get("crisis_endpoint", "Not identified"),
        "preparedness_actions": parse_bullet_list(sections.get("preparedness_actions", "")),
        "plausibility": sections.get("plausibility", "unknown").strip().split()[0].lower(),
        "plausibility_rationale": sections.get("plausibility_rationale", ""),
        "mental_health_relevance": sections.get("mh_relevance", "unknown").strip().split()[0].lower(),
        "mh_relevance_note": sections.get("mh_relevance_note", ""),
    }

    return {
        "scenario_id": branch["id"],
        "branch_type": branch["type"],
        "branch_label": branch["label"],
        "triggering_factors": triggers,
        "causal_chain": causal_chain,
        "scenario": scenario,
    }


# ============================================================================
# FULL ToT GENERATION
# ============================================================================


def generate_scenarios(
    report_text: str,
    silver_labels: dict,
    model,
    tokenizer,
    device: str,
    num_branches: int = NUM_BRANCHES,
) -> dict:
    """
    Run the full Tree-of-Thoughts scenario generation pipeline.

    Args:
        report_text: Original clinical note text.
        silver_labels: Extracted structured factors (Phase 1 output).
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        device: CUDA device string.
        num_branches: Number of scenario branches (2 or 3).

    Returns:
        Dictionary containing relevance assessment + all scenario branches.
    """
    patient_summary = compact_factors(silver_labels)
    # Use first ~1500 chars of the note for context (avoid token overflow)
    note_excerpt = report_text[:1500]

    log.info(f"Phase 2: Running ToT scenario generation ({num_branches} branches) ...")

    # --- Pre-flight: Assess mental health relevance ---
    log.info("Phase 2: Pre-flight mental health relevance assessment ...")
    raw_relevance = call_model(
        model, tokenizer, device,
        system_prompt=SCENARIO_SYSTEM_PROMPT,
        user_prompt=_relevance_assessment_prompt(patient_summary, note_excerpt),
        max_new_tokens=512,
        temperature=0.0,
    )
    relevance = safe_parse_json(raw_relevance)
    log.info(f"Phase 2: MH Relevance = {relevance.get('mh_scenario_relevance', '?')}")

    # --- Run branches ---
    branches_to_run = BRANCHES[:num_branches]
    scenarios = []

    for branch in branches_to_run:
        result = _run_branch(
            branch=branch,
            patient_summary=patient_summary,
            note_excerpt=note_excerpt,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        scenarios.append(result)

    return {
        "phase": "2_tree_of_thoughts_scenario_generation",
        "method": "tree_of_thoughts",
        "num_branches": num_branches,
        "mental_health_relevance_assessment": relevance,
        "patient_factors_summary": {
            "primary_diagnosis": silver_labels.get("primary_mh_diagnosis", {}).get("title", "unknown"),
            "dsm5_category": silver_labels.get("primary_mh_diagnosis", {}).get("dsm5_category", "unknown"),
            "severity": silver_labels.get("severity_level", "unknown"),
            "suicide_ideation": silver_labels.get("suicide_risk_indicators", {}).get("ideation_mentioned", False),
            "substance_use": silver_labels.get("substance_use", {}),
            "social_isolation_risk": silver_labels.get("social_factors", {}).get("social_isolation_risk", False),
        },
        "scenarios": scenarios,
    }
