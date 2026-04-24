"""
Critical Signal Scanner (NEW in V2)
====================================
Rule-based scanner that detects critical safety signals in clinical notes
INDEPENDENT of the model's extraction. This ensures that:

1. Suicide threats are NEVER missed
2. Violence/aggression is NEVER missed
3. Critical signals force a HIGH risk override

This runs on the RAW clinical note text using regex patterns — no model
dependency, no hallucination risk.
"""

import logging
import re
from typing import Dict, List

from .config import CRITICAL_SIGNAL_PATTERNS

log = logging.getLogger(__name__)


# ============================================================================
# SIGNAL DETECTION
# ============================================================================


def scan_critical_signals(note_text: str) -> Dict:
    """
    Scan clinical note text for critical safety signals using regex.

    Returns:
        Dict with:
          - signals_found: bool (any critical signal detected)
          - override_risk_tier: str or None (suggested override)
          - categories: dict mapping category → list of matches
          - alert_summary: str (human-readable alert text)
          - priority_alerts: list of high-priority alert strings
    """
    note_lower = note_text.lower()
    results = {
        "signals_found": False,
        "override_risk_tier": None,
        "categories": {},
        "alert_summary": "",
        "priority_alerts": [],
    }

    all_matches = {}

    for category, patterns in CRITICAL_SIGNAL_PATTERNS.items():
        category_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, note_lower):
                # Get surrounding context (±80 chars)
                start = max(0, match.start() - 80)
                end = min(len(note_text), match.end() + 80)
                context = note_text[start:end].strip()
                # Replace newlines for cleaner display
                context = re.sub(r'\s+', ' ', context)

                category_matches.append({
                    "matched_text": match.group(0),
                    "pattern": pattern,
                    "context": f"...{context}...",
                    "position": match.start(),
                })

        if category_matches:
            # Deduplicate by position (overlapping patterns)
            seen_positions = set()
            unique_matches = []
            for m in category_matches:
                pos_key = m["position"] // 20  # group within 20 chars
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    unique_matches.append(m)
            all_matches[category] = unique_matches

    if all_matches:
        results["signals_found"] = True
        results["categories"] = all_matches

        # Determine override risk tier
        has_suicide = "suicide" in all_matches or "self_harm" in all_matches
        has_violence = "violence" in all_matches

        if has_suicide:
            results["override_risk_tier"] = "HIGH"
            suicide_matches = all_matches.get("suicide", []) + all_matches.get("self_harm", [])
            for m in suicide_matches:
                results["priority_alerts"].append(
                    f"⚠️ SUICIDE/SELF-HARM SIGNAL: '{m['matched_text']}' — {m['context']}"
                )
        if has_violence:
            results["override_risk_tier"] = "HIGH"
            for m in all_matches.get("violence", []):
                results["priority_alerts"].append(
                    f"⚠️ VIOLENCE SIGNAL: '{m['matched_text']}' — {m['context']}"
                )

        # Build summary
        categories_str = ", ".join(
            f"{cat} ({len(matches)} signal{'s' if len(matches) > 1 else ''})"
            for cat, matches in all_matches.items()
        )
        results["alert_summary"] = (
            f"CRITICAL SIGNALS DETECTED: {categories_str}. "
            f"Risk override: {results['override_risk_tier'] or 'none'}."
        )

        log.warning(f"Critical signal scanner: {results['alert_summary']}")
        for alert in results["priority_alerts"]:
            log.warning(f"  {alert}")

    else:
        log.info("Critical signal scanner: No critical signals detected.")

    return results


def augment_silver_labels_with_signals(
    silver_labels: dict,
    critical_signals: Dict,
    note_text: str,
) -> dict:
    """
    Augment silver labels with critical signals that the model may have missed.

    Ensures suicide_risk_indicators and violence_risk_indicators are consistent
    with what's actually in the note.
    """
    if not critical_signals.get("signals_found"):
        return silver_labels

    categories = critical_signals.get("categories", {})

    # ---- Suicide / self-harm signals ----
    suicide_matches = categories.get("suicide", []) + categories.get("self_harm", [])
    if suicide_matches:
        sui = silver_labels.get("suicide_risk_indicators") or {}
        if not isinstance(sui, dict):
            sui = {}

        # Force ideation_mentioned if we found suicide signals
        if not sui.get("ideation_mentioned"):
            sui["ideation_mentioned"] = True
            log.info("Signal scanner: forced suicide_risk_indicators.ideation_mentioned = True")

        # Collect specific threats
        existing_threats = sui.get("specific_threats") or []
        for m in suicide_matches:
            threat_text = m["context"]
            if threat_text not in existing_threats:
                existing_threats.append(threat_text)
        sui["specific_threats"] = existing_threats

        # Upgrade risk level if model said none/low
        current_level = sui.get("risk_level", "none")
        if current_level in ("none", "low"):
            sui["risk_level"] = "moderate"  # at minimum moderate if signal detected
            log.info(f"Signal scanner: upgraded suicide risk_level from '{current_level}' to 'moderate'")

        silver_labels["suicide_risk_indicators"] = sui

    # ---- Violence signals ----
    violence_matches = categories.get("violence", [])
    if violence_matches:
        vio = silver_labels.get("violence_risk_indicators") or {}
        if not isinstance(vio, dict):
            vio = {}

        if not vio.get("aggression_present"):
            vio["aggression_present"] = True
            log.info("Signal scanner: forced violence_risk_indicators.aggression_present = True")

        # Collect specific incidents
        existing_incidents = vio.get("specific_incidents") or []
        for m in violence_matches:
            incident_text = m["context"]
            if incident_text not in existing_incidents:
                existing_incidents.append(incident_text)
        vio["specific_incidents"] = existing_incidents

        # Upgrade risk level
        current_level = vio.get("risk_level", "none")
        if current_level in ("none", "low"):
            vio["risk_level"] = "moderate"
            log.info(f"Signal scanner: upgraded violence risk_level from '{current_level}' to 'moderate'")

        silver_labels["violence_risk_indicators"] = vio

    return silver_labels
