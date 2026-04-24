"""
Phase 3: Patient Report Generator
===================================
Generates a concise, clinician-readable report that synthesizes:
  1. Patient demographic and clinical overview (from silver labels)
  2. Key risk factors and protective factors
  3. Tree-of-Thoughts scenario summaries with plausibility ratings
  4. Actionable preparedness recommendations

The report is designed to serve as a "second reader" briefing — a
structured summary that clinicians and psychology students can use
to anticipate and prepare for extreme negative patient trajectories.
"""

import logging
from datetime import datetime

from .config import REPORT_MAX_NEW_TOKENS, REPORT_TEMPERATURE
from .model_loader import call_model
from .utils import compact_factors, parse_marked_sections, parse_bullet_list

log = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT — Report Generation
# ============================================================================

REPORT_SYSTEM_PROMPT = """\
You are a senior clinical report writer specialising in mental health \
patient briefings for clinical preparedness.

YOUR TASK: Generate a concise, professional clinical preparedness report \
that synthesizes patient data and scenario analyses into an actionable briefing.

REPORT STYLE:
- Clinical and professional tone
- Concise but comprehensive
- Actionable — end with specific recommendations
- Honest about uncertainty — if mental health relevance is low, say so
- NEVER fabricate information not present in the input data
- Use EXACTLY the section headers provided
- Do NOT wrap output in JSON or markdown fences"""


# ============================================================================
# REPORT PROMPT
# ============================================================================


def _build_report_prompt(
    patient_summary: str,
    silver_labels: dict,
    scenarios_result: dict,
    report_text_excerpt: str,
) -> str:
    """Build the report generation prompt (plain text output)."""

    # Summarize scenarios compactly as plain text
    scenario_lines = []
    for s in scenarios_result.get("scenarios", []):
        scenario_obj = s.get("scenario", {})
        label = s.get("branch_label", "?")
        plaus = scenario_obj.get("plausibility", "?")
        mh_rel = scenario_obj.get("mental_health_relevance", "?")
        crisis = scenario_obj.get("crisis_endpoint", "?")
        narrative_preview = scenario_obj.get("narrative", "")[:200]
        actions = scenario_obj.get("preparedness_actions", [])[:3]
        scenario_lines.append(
            f"- {label}\n"
            f"  Plausibility: {plaus} | MH Relevance: {mh_rel}\n"
            f"  Crisis endpoint: {crisis}\n"
            f"  Narrative: {narrative_preview}...\n"
            f"  Actions: {'; '.join(actions) if actions else 'None'}"
        )

    relevance = scenarios_result.get("mental_health_relevance_assessment", {})

    return (
        "### PATIENT CLINICAL FACTORS ###\n"
        f"{patient_summary}\n\n"
        "### MENTAL HEALTH RELEVANCE ###\n"
        f"Primary is psychiatric: {relevance.get('primary_is_psychiatric', '?')}\n"
        f"Has psych history: {relevance.get('has_psych_history', '?')}\n"
        f"MH relevance: {relevance.get('mh_scenario_relevance', '?')}\n"
        f"Note: {relevance.get('explanation', '')}\n\n"
        "### SCENARIO SUMMARIES ###\n"
        + '\n'.join(scenario_lines) + "\n\n"
        "### CLINICAL NOTE (Excerpt) ###\n"
        f"{report_text_excerpt}\n\n"
        "### TASK ###\n"
        "Write a clinical preparedness report using EXACTLY these sections:\n\n"
        "PATIENT OVERVIEW:\n"
        "(3-5 sentences: who is this patient, primary diagnosis, "
        "admission reason, key features. Skip redacted names/IDs.)\n\n"
        "MENTAL HEALTH STATUS:\n"
        "(2-3 sentences on current MH state. If primary condition is "
        "non-psychiatric, note that and describe only genuine MH implications.)\n\n"
        "KEY RISK FACTORS:\n"
        "- factor 1\n- factor 2\n"
        "(list most significant risk factors from the data)\n\n"
        "PROTECTIVE FACTORS:\n"
        "- factor 1\n- factor 2\n"
        "(list factors that reduce risk)\n\n"
        "SCENARIO BRIEFING:\n"
        "(For each of the 3 pathways, write 2-3 sentences: pathway name, "
        "plausibility, key concern, top action. If LOW relevance, say why.)\n\n"
        "PRIORITY ACTIONS:\n"
        "1. action\n2. action\n"
        "(Top 5 actionable clinical recommendations for THIS patient)\n\n"
        "RISK TIER: HIGH or MODERATE or LOW\n\n"
        "RISK JUSTIFICATION:\n"
        "(1-2 sentences explaining the risk rating)"
    )


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(
    report_text: str,
    silver_labels: dict,
    scenarios_result: dict,
    model,
    tokenizer,
    device: str,
) -> dict:
    """
    Generate the final patient preparedness report.

    Returns a dict parsed from plain-text section markers.
    """
    log.info("Phase 3: Generating patient preparedness report ...")

    patient_summary = compact_factors(silver_labels)
    note_excerpt = report_text[:1200]

    prompt = _build_report_prompt(
        patient_summary, silver_labels, scenarios_result, note_excerpt
    )

    raw_output = call_model(
        model, tokenizer, device,
        system_prompt=REPORT_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_new_tokens=REPORT_MAX_NEW_TOKENS,
        temperature=REPORT_TEMPERATURE,
    )

    log.info(f"Phase 3: Report generated ({len(raw_output)} chars)")

    # Parse plain-text sections into a dict
    _REPORT_MARKERS = [
        "PATIENT OVERVIEW", "MENTAL HEALTH STATUS",
        "KEY RISK FACTORS", "PROTECTIVE FACTORS",
        "SCENARIO BRIEFING", "PRIORITY ACTIONS",
        "RISK TIER", "RISK JUSTIFICATION",
    ]
    sections = parse_marked_sections(raw_output, _REPORT_MARKERS)

    report = {
        "patient_overview": sections.get("patient_overview", ""),
        "mental_health_status": sections.get("mental_health_status", ""),
        "key_risk_factors": parse_bullet_list(sections.get("key_risk_factors", "")),
        "protective_factors": parse_bullet_list(sections.get("protective_factors", "")),
        "scenario_briefing": sections.get("scenario_briefing", ""),
        "priority_actions": parse_bullet_list(sections.get("priority_actions", "")),
        "overall_risk_tier": sections.get("risk_tier", "UNKNOWN").strip().split("\n")[0].strip(),
        "risk_tier_justification": sections.get("risk_justification", ""),
    }

    # If parsing found nothing, store the raw text so it still gets displayed
    if not any(v for k, v in report.items() if k != "overall_risk_tier"):
        log.warning("Phase 3: Section parsing found no content; storing raw output.")
        report["patient_overview"] = raw_output[:2000]
        report["_raw_text"] = raw_output[:3000]

    return report


# ============================================================================
# FORMATTED DISPLAY
# ============================================================================


def format_report_text(
    report: dict,
    silver_labels: dict,
    scenarios_result: dict,
) -> str:
    """
    Format the report as a human-readable text document.

    Returns a string suitable for printing to console or saving as .txt.
    """
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  CLINICAL PREPAREDNESS REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # Diagnosis header
    dx = silver_labels.get("primary_mh_diagnosis", {})
    if isinstance(dx, dict):
        dx_title = dx.get("title", "Unknown")
        dx_code = dx.get("icd_code", "")
        dx_cat = dx.get("dsm5_category", "Unknown")
    else:
        dx_title, dx_code, dx_cat = "Unknown", "", "Unknown"
    lines.append(f"  Primary Dx: {dx_title} ({dx_code})")
    lines.append(f"  DSM-5 Category: {dx_cat}")
    lines.append(f"  Severity: {silver_labels.get('severity_level', 'Unknown')}")

    risk_tier = report.get("overall_risk_tier", "?")
    lines.append(f"  Overall MH Risk: {risk_tier}")
    lines.append(sep)

    # Patient Overview
    lines.append("\n1. PATIENT OVERVIEW")
    lines.append("-" * 40)
    lines.append(report.get("patient_overview", "Not available."))

    # MH Status
    lines.append("\n2. MENTAL HEALTH STATUS")
    lines.append("-" * 40)
    lines.append(report.get("mental_health_status", "Not available."))

    # Risk Factors
    lines.append("\n3. KEY RISK FACTORS")
    lines.append("-" * 40)
    for rf in report.get("key_risk_factors", []):
        lines.append(f"  • {rf}")

    # Protective Factors
    lines.append("\n4. PROTECTIVE FACTORS")
    lines.append("-" * 40)
    for pf in report.get("protective_factors", []):
        lines.append(f"  • {pf}")

    # Scenario Briefing
    lines.append("\n5. COUNTERFACTUAL SCENARIO BRIEFING")
    lines.append("-" * 40)

    briefing = report.get("scenario_briefing", "")
    if isinstance(briefing, str) and briefing:
        lines.append(briefing)
    elif isinstance(briefing, list):
        for i, sb in enumerate(briefing, 1):
            if isinstance(sb, dict):
                lines.append(f"\n  Pathway {i}: {sb.get('pathway', '?')}")
                lines.append(f"  Plausibility: {sb.get('plausibility', '?')}")
                lines.append(f"  Summary: {sb.get('summary', '?')}")
                lines.append(f"  Top Action: {sb.get('top_action', '?')}")
            elif isinstance(sb, str):
                lines.append(f"  {sb}")

    # Also show detailed scenario data from Phase 2
    lines.append("\n  --- Detailed Scenario Data ---")
    for s in scenarios_result.get("scenarios", []):
        lines.append(f"\n  [{s.get('scenario_id', '?')}] {s.get('branch_label', '?')}")

        # Triggers
        triggers = s.get("triggering_factors", [])
        if isinstance(triggers, list):
            for t in triggers:
                if isinstance(t, dict):
                    lines.append(
                        f"    Trigger: {t.get('factor', '?')} "
                        f"(MH relevance: {t.get('mh_relevance', '?')})"
                    )
                    lines.append(f"      → {t.get('reason', t.get('evidence', ''))}")

        # Causal chain
        chain = s.get("causal_chain", [])
        if isinstance(chain, list) and chain:
            lines.append("    Causal Chain:")
            for step in chain:
                if isinstance(step, dict):
                    lines.append(
                        f"      {step.get('step', '?')}. {step.get('event', '')} "
                        f"[{step.get('mechanism', '')}]"
                    )

        # Narrative
        scenario_obj = s.get("scenario", {})
        if isinstance(scenario_obj, dict) and not scenario_obj.get("_parse_failed"):
            narrative = scenario_obj.get("narrative", "")
            if narrative:
                lines.append(f"    Narrative: {narrative[:500]}")
            crisis = scenario_obj.get("crisis_endpoint", "")
            if crisis:
                lines.append(f"    Crisis Endpoint: {crisis}")
            plaus = scenario_obj.get("plausibility", "?")
            p_rat = scenario_obj.get("plausibility_rationale", "")
            lines.append(f"    Plausibility: {plaus}{' — ' + p_rat if p_rat else ''}")
            mh_rel = scenario_obj.get("mental_health_relevance", "?")
            mh_note = scenario_obj.get("mh_relevance_note", "")
            lines.append(f"    MH Relevance: {mh_rel}{' — ' + mh_note if mh_note else ''}")
            # Warning signs
            wsigns = scenario_obj.get("warning_signs", [])
            if wsigns:
                lines.append("    Warning Signs:")
                for ws in wsigns[:5]:
                    lines.append(f"      - {ws}")
            # Actions
            actions = scenario_obj.get("preparedness_actions", [])
            if actions:
                lines.append("    Preparedness Actions:")
                for a in actions[:5]:
                    lines.append(f"      - {a}")

    # Priority Actions
    lines.append("\n6. PRIORITY PREPAREDNESS ACTIONS")
    lines.append("-" * 40)
    for i, action in enumerate(report.get("priority_actions", []), 1):
        lines.append(f"  {i}. {action}")

    # Risk Tier
    lines.append(f"\n7. OVERALL RISK TIER: {risk_tier}")
    lines.append(f"   {report.get('risk_tier_justification', '')}")

    lines.append("\n" + sep)
    lines.append("  END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)
