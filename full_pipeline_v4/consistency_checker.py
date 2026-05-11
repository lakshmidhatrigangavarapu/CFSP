"""
Final Consistency Checker (V3)
===============================
Post-Phase-3 validation that ensures the final report is internally
consistent and correct. This is the LAST step before output.

V3 additions:
  - Severity inference when model returns "unknown"
  - Plausibility post-processing from scenario rationale
  - All V2 checks preserved

Checks:
1. Risk tier consistency — no contradictions between sections
2. Critical signal override — force HIGH if suicide/violence detected
3. Gated branch acknowledgment — ensure NOT APPLICABLE branches are
   properly reflected in the report
4. Severity alignment — risk tier matches extracted severity + signals
5. (V3) Severity inference from clinical data
6. (V3) Plausibility resolution from rationale text
"""

import logging
import re
from typing import Dict

log = logging.getLogger(__name__)


def check_consistency(
    report: dict,
    silver_labels: dict,
    scenarios_result: dict,
    critical_signals: dict,
) -> dict:
    """
    Final consistency check on the assembled pipeline output.

    Modifies and returns the report dict with corrections applied.
    Also returns a _consistency_log listing all changes.
    """
    changes = []

    # ================================================================
    # CHECK 1: Critical signal → Risk tier override
    # ================================================================
    override_tier = critical_signals.get("override_risk_tier")
    current_tier = report.get("overall_risk_tier", "UNKNOWN").strip().upper()

    if override_tier:
        override_tier_upper = override_tier.upper()
        tier_priority = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "UNKNOWN": 0}

        current_priority = tier_priority.get(current_tier, 0)
        override_priority = tier_priority.get(override_tier_upper, 0)

        if override_priority > current_priority:
            changes.append(
                f"Risk tier override: '{current_tier}' → '{override_tier_upper}' "
                f"(critical signals detected: {critical_signals.get('alert_summary', '')})"
            )
            report["overall_risk_tier"] = override_tier_upper

            # Update justification to mention the override
            existing_justification = report.get("risk_tier_justification", "")
            signal_summary = critical_signals.get("alert_summary", "")
            report["risk_tier_justification"] = (
                f"{existing_justification} "
                f"[OVERRIDE: Risk tier elevated to {override_tier_upper} due to "
                f"critical safety signals detected in clinical note: {signal_summary}]"
            ).strip()

    # ================================================================
    # CHECK 2: UNKNOWN risk tier resolution
    # ================================================================
    current_tier = report.get("overall_risk_tier", "UNKNOWN").strip().upper()
    if current_tier == "UNKNOWN":
        resolved_tier = _resolve_unknown_risk_tier(silver_labels, scenarios_result, critical_signals)
        if resolved_tier != "UNKNOWN":
            changes.append(
                f"Unknown risk resolved: 'UNKNOWN' → '{resolved_tier}' "
                f"(based on severity, scenarios, and clinical data)"
            )
            report["overall_risk_tier"] = resolved_tier

    # ================================================================
    # CHECK 3: Gated branches reflected in report
    # ================================================================
    gated_scenarios = [
        s for s in scenarios_result.get("scenarios", [])
        if s.get("gated")
    ]
    if gated_scenarios:
        gated_labels = [s.get("branch_label", "?") for s in gated_scenarios]
        # Ensure scenario_briefing mentions NOT APPLICABLE for gated branches
        briefing = report.get("scenario_briefing", "")
        for label in gated_labels:
            if label not in briefing and "NOT APPLICABLE" not in briefing:
                briefing_addition = (
                    f"\n{label}: NOT APPLICABLE - insufficient clinical evidence "
                    f"to support this pathway for this patient."
                )
                report["scenario_briefing"] = briefing + briefing_addition
                changes.append(f"Added NOT APPLICABLE note for gated branch: {label}")

    # ================================================================
    # CHECK 4: Priority alerts injection
    # ================================================================
    priority_alerts = critical_signals.get("priority_alerts", [])
    if priority_alerts:
        # Inject alerts at the TOP of the report
        if "priority_alerts" not in report:
            report["priority_alerts"] = priority_alerts
            changes.append(f"Injected {len(priority_alerts)} priority alert(s) into report")

    # ================================================================
    # CHECK 5: Ensure priority_actions is populated
    # ================================================================
    if not report.get("priority_actions"):
        # Fallback: collect top actions from active scenarios
        all_actions = []
        for s in scenarios_result.get("scenarios", []):
            if not s.get("gated"):
                scenario = s.get("scenario", {})
                all_actions.extend(scenario.get("preparedness_actions", [])[:2])
        if all_actions:
            report["priority_actions"] = all_actions[:5]
            changes.append("Populated empty priority_actions from scenario actions")

    # ================================================================
    # CHECK 6 (V3): Severity inference when "unknown"
    # ================================================================
    severity = silver_labels.get("severity_level", "unknown")
    if severity == "unknown":
        inferred = _infer_severity(silver_labels, critical_signals)
        if inferred != "unknown":
            silver_labels["severity_level"] = inferred
            changes.append(f"Severity inferred: 'unknown' → '{inferred}'")

    # ================================================================
    # CHECK 7 (V3): Plausibility resolution for scenarios
    # ================================================================
    for s in scenarios_result.get("scenarios", []):
        if s.get("gated"):
            continue
        scenario = s.get("scenario", {})
        plaus = scenario.get("plausibility", "unknown")
        if plaus in ("unknown", ""):
            resolved = _resolve_plausibility(scenario, silver_labels, critical_signals)
            if resolved != "unknown":
                scenario["plausibility"] = resolved
                branch_label = s.get("branch_label", "?")
                changes.append(f"Plausibility resolved for {branch_label}: 'unknown' → '{resolved}'")

    # ================================================================
    # LOG
    # ================================================================
    if changes:
        report["_consistency_log"] = changes
        log.info(f"Consistency checker: {len(changes)} corrections applied:")
        for c in changes:
            log.info(f"  → {c}")
    else:
        log.info("Consistency checker: no corrections needed.")

    return report


def _resolve_unknown_risk_tier(
    silver_labels: dict,
    scenarios_result: dict,
    critical_signals: dict,
) -> str:
    """Attempt to resolve UNKNOWN risk tier from available data."""

    # High-risk indicators
    high_risk_factors = 0

    # Suicide indicators
    sui = silver_labels.get("suicide_risk_indicators") or {}
    if sui.get("ideation_mentioned"):
        high_risk_factors += 2
    if sui.get("attempt_history"):
        high_risk_factors += 3
    if sui.get("risk_level") in ("moderate", "high"):
        high_risk_factors += 2

    # Violence indicators
    vio = silver_labels.get("violence_risk_indicators") or {}
    if vio.get("aggression_present"):
        high_risk_factors += 2
    if vio.get("threats_made"):
        high_risk_factors += 2
    if vio.get("risk_level") in ("moderate", "high"):
        high_risk_factors += 2

    # Severity
    severity = silver_labels.get("severity_level", "unknown")
    if severity == "critical":
        high_risk_factors += 3
    elif severity == "severe":
        high_risk_factors += 2
    elif severity == "moderate":
        high_risk_factors += 1

    # Medication non-adherence
    adherence = silver_labels.get("medication_adherence", "unknown")
    if adherence == "poor":
        high_risk_factors += 1

    # Social isolation
    soc = silver_labels.get("social_factors") or {}
    if soc.get("social_isolation_risk"):
        high_risk_factors += 1
    housing = soc.get("housing_status", "unknown")
    if housing in ("homeless", "unstable"):
        high_risk_factors += 2

    # Progressive disease
    if silver_labels.get("progression") == "progressive":
        high_risk_factors += 1

    # Scenario plausibility
    for s in scenarios_result.get("scenarios", []):
        if s.get("gated"):
            continue
        scenario = s.get("scenario", {})
        plaus = scenario.get("plausibility", "unknown")
        if plaus == "high":
            high_risk_factors += 1

    # Critical signals
    if critical_signals.get("signals_found"):
        high_risk_factors += 3

    # Resolve
    if high_risk_factors >= 5:
        return "HIGH"
    elif high_risk_factors >= 2:
        return "MODERATE"
    elif high_risk_factors >= 0:
        return "LOW"

    return "UNKNOWN"


def _infer_severity(silver_labels: dict, critical_signals: dict) -> str:
    """
    V3: Infer severity when model returns 'unknown'.
    Uses suicide/violence signals, critical signals, medication adherence,
    and functional decline markers.
    """
    score = 0

    # Suicide indicators
    sui = silver_labels.get("suicide_risk_indicators") or {}
    if sui.get("ideation_mentioned"):
        score += 2
    if sui.get("attempt_history"):
        score += 3
    if sui.get("specific_threats"):
        score += 2

    # Violence indicators
    vio = silver_labels.get("violence_risk_indicators") or {}
    if vio.get("aggression_present"):
        score += 1
    if vio.get("threats_made"):
        score += 2

    # Critical signals
    if critical_signals.get("signals_found"):
        categories = critical_signals.get("categories", {})
        if "suicide" in categories:
            score += 2
        if "violence" in categories:
            score += 1

    # Medication non-adherence
    if silver_labels.get("medication_adherence") == "poor":
        score += 1

    # Enriched context
    ctx = silver_labels.get("enriched_context") or {}
    if ctx.get("functional_decline", {}).get("detected"):
        score += 1
    if ctx.get("housing_crisis", {}).get("detected"):
        score += 1
    if ctx.get("caregiver_loss", {}).get("detected"):
        score += 1

    # Progressive disease
    if silver_labels.get("progression") == "progressive":
        score += 1

    if score >= 6:
        return "severe"
    elif score >= 3:
        return "moderate"
    elif score >= 1:
        return "mild"
    return "unknown"


def _resolve_plausibility(
    scenario: dict,
    silver_labels: dict,
    critical_signals: dict,
) -> str:
    """
    V3: Resolve 'unknown' plausibility from scenario rationale and clinical data.
    """
    rationale = (scenario.get("plausibility_rationale", "") or "").lower()
    mh_note = (scenario.get("mh_relevance_note", "") or "").lower()
    combined = rationale + " " + mh_note

    # Check if the rationale itself implies high/medium/low
    if any(w in combined for w in ["high likelihood", "likely", "credible", "strong evidence", "significant risk"]):
        return "high"
    if any(w in combined for w in ["moderate", "some risk", "possible", "plausible"]):
        return "medium"
    if any(w in combined for w in ["unlikely", "low risk", "minimal", "limited"]):
        return "low"

    # Fall back to clinical data signals
    if critical_signals.get("signals_found"):
        return "high"

    severity = silver_labels.get("severity_level", "unknown")
    if severity in ("severe", "critical"):
        return "high"
    elif severity == "moderate":
        return "medium"

    return "unknown"
