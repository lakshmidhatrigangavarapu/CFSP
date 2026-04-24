"""
Phase 1.55: Context Enricher (NEW in V3)
==========================================
Scans the raw clinical note for contextual information that the finetuned
model's fixed schema cannot capture. Injects structured context into the
silver_labels dict so that compact_factors() can pass it to the scenario
generator.

Extracts:
  1. Housing crisis (eviction, homelessness, shelter, auctioned)
  2. Caregiver/relationship loss (partner leaving, divorce, death of caregiver)
  3. Trauma history (sexual abuse, physical abuse, childhood trauma, DV)
  4. Weight change (significant weight loss/gain, self-neglect)
  5. Functional decline (dysarthria, can't leave house, ADL impairment)
  6. Clinician prognosis statements (guarded, poor, high risk of future X)
  7. Legal status (voluntary, involuntary, conditional voluntary, court-ordered)

This module does NOT touch prompts. It only enriches the structured data
that gets passed to downstream phases via compact_factors().
"""

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ============================================================================
# CONTEXT PATTERN DEFINITIONS
# ============================================================================

CONTEXT_PATTERNS = {
    "housing_crisis": {
        "patterns": [
            (r'homeless\w*', "homelessness"),
            (r'(?:will\s+be|facing|at\s+risk\s+of)\s+homeless', "imminent homelessness"),
            (r'evict\w*', "eviction"),
            (r'auction\w*\s+(?:off|away)?', "property auctioned"),
            (r'(?:apt|apartment|house|home)\s+(?:has\s+been\s+)?(?:auction|sold|lost|foreclos)', "housing loss"),
            (r'shelter', "shelter placement"),
            (r'(?:no|lacks?|without)\s+(?:stable\s+)?(?:housing|home|residence)', "unstable housing"),
            (r'(?:discharg\w+|going|sent|transfer\w*)\s+to\s+(?:a\s+)?(?:assisted\s+living|nursing|rehab|facility|group\s+home)', "facility placement"),
        ],
        "category": "social",
    },
    "caregiver_loss": {
        "patterns": [
            (r'(?:boyfriend|girlfriend|partner|spouse|husband|wife|caregiver)\s+(?:has\s+)?(?:decided\s+to\s+)?(?:leav|left|abandon|gone)', "caregiver leaving"),
            (r'(?:no\s+longer|not)\s+able\s+to\s+(?:handle|care|manage|cope)', "caregiver burnout"),
            (r'(?:relationship|marriage)\s+(?:break\w*|end\w*|dissolv)', "relationship breakdown"),
            (r'divorc\w*', "divorce"),
            (r'(?:primary\s+)?(?:caregiver|support\s+person)\s+(?:died|deceased|passed|unavailable)', "caregiver death"),
            (r'(?:family|support)\s+(?:no\s+longer|unable|refuse)', "loss of family support"),
        ],
        "category": "social",
    },
    "trauma_history": {
        "patterns": [
            (r'(?:sexual\w*)\s+(?:molest|abus|assault|trauma)', "sexual abuse/assault"),
            (r'(?:physical\w*)\s+(?:abus|assault)', "physical abuse"),
            (r'(?:childhood|early)\s+(?:trauma|abuse|neglect)', "childhood trauma"),
            (r'(?:domestic\s+)?violence\s+(?:victim|survivor|history)', "domestic violence"),
            (r'(?:endorses?|reports?|history\s+of)\s+(?:being\s+)?(?:molest|abus|assault|rape)', "disclosed abuse"),
            (r'(?:trauma|PTSD|post[- ]?traumatic)', "trauma/PTSD"),
        ],
        "category": "psychiatric",
    },
    "weight_change": {
        "patterns": [
            (r'(?:lost|loss\s+of)\s+(?:weight\s*)?(?:\(?\d+\s*(?:lbs?|pounds?|kg)\)?)', "significant weight loss"),
            (r'(?:weight\s+(?:loss|lost))\s*(?:\(?\d+\s*(?:lbs?|pounds?|kg)\)?)?', "weight loss"),
            (r'(?:very\s+)?(?:thin|emaciat|cachecti|underweight|malnourish)', "thin/emaciated"),
            (r'(?:gained?|gain\s+of)\s+(?:\d+\s*(?:lbs?|pounds?|kg))', "weight gain"),
            (r'(?:poor|decreased|reduced)\s+(?:oral\s+)?(?:intake|nutrition|appetite)', "poor nutrition"),
            (r'(?:self[- ]?neglect|unable\s+to\s+(?:feed|eat|care\s+for))', "self-neglect"),
        ],
        "category": "medical",
    },
    "functional_decline": {
        "patterns": [
            (r'dysarthri[ca]', "dysarthria (speech impairment)"),
            (r'(?:has\s+)?not\s+left\s+(?:the\s+)?(?:house|home|apartment|room)\s+(?:in|for)\s+(?:\w+\s+)?(?:year|month|week)', "prolonged homebound"),
            (r'(?:difficulty|unable|cannot|can\'t)\s+(?:car\w+\s+for\s+(?:self|her|him))', "self-care difficulty"),
            (r'(?:difficulty|unable|cannot)\s+(?:walk|ambulate|move|dress|bathe|toilet|groom)', "ADL impairment"),
            (r'ataxic\s+(?:gait)?', "ataxic gait"),
            (r'wide[- ]?based\s+(?:gait|stance)', "wide-based gait"),
            (r'monosyllab\w+\s+(?:speech|phrase|response)', "monosyllabic speech"),
            (r'speech\s+latency', "speech latency"),
            (r'(?:cognitive|mental)\s+(?:decline|impairment|deteriorat)', "cognitive decline"),
        ],
        "category": "medical",
    },
    "clinician_prognosis": {
        "patterns": [
            (r'prognosis[:\s]+(?:guarded|poor|grave|uncertain|limited)', "clinician prognosis statement"),
            (r'(?:high|elevated|significant)\s+risk\s+of\s+(?:future\s+)?(?:decompensat|relapse|deteriorat|recurren)', "high future decompensation risk"),
            (r'progressive\s+(?:neuro)?degenerative\s+(?:condition|disease|disorder)', "progressive neurodegenerative condition"),
            (r'(?:associated\s+w(?:ith)?|linked\s+to)\s+(?:mood\s+swings|psychosis|cognitive\s+decline)', "disease-associated psychiatric risk"),
            (r'later\s+stages', "late-stage disease reference"),
        ],
        "category": "clinical",
    },
    "legal_status": {
        "patterns": [
            (r'(?:remained?\s+on\s+)?(?:a\s+)?(?:conditional\s+voluntary|CV)\b', "conditional voluntary"),
            (r'(?:involuntary|committed|sectioned)\s+(?:hold|commitment|admission)', "involuntary hold"),
            (r'(?:voluntary|vol)\s+(?:admission|status)', "voluntary status"),
            (r'(?:court[- ]?order|mandated|forensic)', "court-ordered"),
            (r'(?:5150|5250|302|pink\s+slip)', "emergency psychiatric hold"),
        ],
        "category": "legal",
    },
}


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================


def _extract_context_matches(note_text: str) -> Dict[str, List[dict]]:
    """
    Scan clinical note for all context patterns.
    Returns dict mapping context_type → list of matches.
    """
    note_lower = note_text.lower()
    results = {}

    for context_type, config in CONTEXT_PATTERNS.items():
        matches = []
        seen_positions = set()

        for pattern, label in config["patterns"]:
            for match in re.finditer(pattern, note_lower):
                # Deduplicate by position (within 30 chars)
                pos_key = match.start() // 30
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                # Get surrounding context (±100 chars)
                start = max(0, match.start() - 100)
                end = min(len(note_text), match.end() + 100)
                context = note_text[start:end].strip()
                context = re.sub(r'\s+', ' ', context)

                matches.append({
                    "label": label,
                    "matched_text": match.group(0),
                    "context": context,
                    "position": match.start(),
                })

        if matches:
            results[context_type] = matches

    return results


def _extract_weight_amount(note_text: str) -> Optional[str]:
    """Try to extract specific weight loss/gain amount."""
    patterns = [
        r'(?:lost|loss\s+of)\s+(?:weight\s*)?(?:\(?(\d+)\s*(?:lbs?|pounds?)\)?)',
        r'weight\s+(?:loss|lost)\s*(?:\(?(\d+)\s*(?:lbs?|pounds?)\)?)',
        r'(\d+)\s*(?:lbs?|pounds?)\s+(?:weight\s+)?(?:loss|lost)',
    ]
    for pattern in patterns:
        m = re.search(pattern, note_text.lower())
        if m and m.group(1):
            return f"{m.group(1)} lbs"
    return None


def _extract_homebound_duration(note_text: str) -> Optional[str]:
    """Try to extract how long patient has been homebound."""
    m = re.search(
        r'not\s+left\s+(?:the\s+)?(?:house|home)\s+(?:in|for)\s+(?:over\s+)?(\w+\s+(?:year|month|week)s?)',
        note_text.lower()
    )
    if m:
        return m.group(1)
    return None


# ============================================================================
# MAIN ENRICHER
# ============================================================================


def enrich_context(
    silver_labels: dict,
    note_text: str,
) -> dict:
    """
    Enrich silver labels with contextual information extracted from the
    raw clinical note.

    Adds an 'enriched_context' dict to silver_labels containing structured
    context that the model's schema couldn't capture.

    Args:
        silver_labels: Silver labels from Phase 1 + normalization.
        note_text: Raw clinical note text (cleaned).

    Returns:
        silver_labels with 'enriched_context' added.
    """
    if not silver_labels or silver_labels.get("_parse_failed"):
        return silver_labels

    context_matches = _extract_context_matches(note_text)
    enrichment_log = []
    enriched = {}

    # --- Housing ---
    housing_matches = context_matches.get("housing_crisis", [])
    if housing_matches:
        labels = [m["label"] for m in housing_matches]
        best_context = housing_matches[0]["context"]
        enriched["housing_crisis"] = {
            "detected": True,
            "details": labels,
            "evidence": best_context,
        }
        enrichment_log.append(f"Housing crisis detected: {', '.join(labels)}")

        # Also update social_factors if housing is currently unknown
        soc = silver_labels.get("social_factors") or {}
        if not isinstance(soc, dict):
            soc = {}
        if soc.get("housing_status", "unknown") == "unknown":
            # Determine housing status from what we found
            if any("homeless" in l for l in labels):
                soc["housing_status"] = "homeless"
            elif any("facility" in l for l in labels):
                soc["housing_status"] = "facility_placement"
            else:
                soc["housing_status"] = "unstable"
            silver_labels["social_factors"] = soc
            enrichment_log.append(f"Updated housing_status to '{soc['housing_status']}'")

    # --- Caregiver loss ---
    caregiver_matches = context_matches.get("caregiver_loss", [])
    if caregiver_matches:
        labels = [m["label"] for m in caregiver_matches]
        best_context = caregiver_matches[0]["context"]
        enriched["caregiver_loss"] = {
            "detected": True,
            "details": labels,
            "evidence": best_context,
        }
        enrichment_log.append(f"Caregiver/relationship loss detected: {', '.join(labels)}")

        # Update social isolation risk
        soc = silver_labels.get("social_factors") or {}
        if not isinstance(soc, dict):
            soc = {}
        if not soc.get("social_isolation_risk"):
            soc["social_isolation_risk"] = True
            silver_labels["social_factors"] = soc
            enrichment_log.append("Updated social_isolation_risk to True (caregiver loss)")

    # --- Trauma history ---
    trauma_matches = context_matches.get("trauma_history", [])
    if trauma_matches:
        labels = [m["label"] for m in trauma_matches]
        best_context = trauma_matches[0]["context"]
        enriched["trauma_history"] = {
            "detected": True,
            "details": labels,
            "evidence": best_context,
        }
        enrichment_log.append(f"Trauma history detected: {', '.join(labels)}")

    # --- Weight change ---
    weight_matches = context_matches.get("weight_change", [])
    if weight_matches:
        labels = [m["label"] for m in weight_matches]
        amount = _extract_weight_amount(note_text)
        enriched["weight_change"] = {
            "detected": True,
            "details": labels,
            "amount": amount,
            "evidence": weight_matches[0]["context"],
        }
        enrichment_log.append(
            f"Weight change detected: {', '.join(labels)}"
            + (f" ({amount})" if amount else "")
        )

    # --- Functional decline ---
    functional_matches = context_matches.get("functional_decline", [])
    if functional_matches:
        labels = [m["label"] for m in functional_matches]
        enriched["functional_decline"] = {
            "detected": True,
            "details": labels,
            "evidence": functional_matches[0]["context"],
        }
        enrichment_log.append(f"Functional decline detected: {', '.join(labels)}")

        # Extract specific duration if homebound
        duration = _extract_homebound_duration(note_text)
        if duration:
            enriched["functional_decline"]["homebound_duration"] = duration
            enrichment_log.append(f"Homebound duration: {duration}")

    # --- Clinician prognosis ---
    prognosis_matches = context_matches.get("clinician_prognosis", [])
    if prognosis_matches:
        labels = [m["label"] for m in prognosis_matches]
        # Get the most informative context
        best_match = max(prognosis_matches, key=lambda m: len(m["context"]))
        enriched["clinician_prognosis"] = {
            "detected": True,
            "details": labels,
            "statement": best_match["context"],
        }
        enrichment_log.append(f"Clinician prognosis: {', '.join(labels)}")

    # --- Legal status ---
    legal_matches = context_matches.get("legal_status", [])
    if legal_matches:
        labels = [m["label"] for m in legal_matches]
        enriched["legal_status"] = {
            "detected": True,
            "details": labels,
            "evidence": legal_matches[0]["context"],
        }
        enrichment_log.append(f"Legal status detected: {', '.join(labels)}")

    # --- Store enrichment ---
    if enriched:
        silver_labels["enriched_context"] = enriched
        silver_labels["_enrichment_log"] = enrichment_log
        log.info(f"Context enricher found {len(enriched)} context categories:")
        for entry in enrichment_log:
            log.info(f"  + {entry}")
    else:
        silver_labels["enriched_context"] = {}
        silver_labels["_enrichment_log"] = []
        log.info("Context enricher: no additional context found.")

    return silver_labels
