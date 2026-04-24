"""
Branch Gating (NEW in V2)
==========================
Determines which ToT scenario branches should run based on actual
clinical evidence in the silver labels.

Rules:
  - Branch A (Clinical Deterioration): ALWAYS runs
  - Branch B (Substance Escalation): ONLY if substance use evidence exists
  - Branch C (Social/Environmental): ALWAYS runs

When a branch is gated out, it's marked "NOT APPLICABLE" with a reason,
instead of being silently skipped. This is important for the report.
"""

import logging
from typing import List, Tuple

from .config import BRANCHES

log = logging.getLogger(__name__)


def evaluate_branch_gates(
    silver_labels: dict,
    critical_signals: dict,
) -> List[dict]:
    """
    Evaluate which branches should run based on evidence.

    Returns:
        List of branch dicts, each augmented with:
          - "gated": bool (True = should NOT run)
          - "gate_reason": str (why it was gated out, or why it should run)
    """
    results = []

    for branch in BRANCHES:
        branch_result = dict(branch)  # copy
        gate_condition = branch.get("gate_condition", "always")

        if gate_condition == "always":
            branch_result["gated"] = False
            branch_result["gate_reason"] = "Core pathway — always evaluated"

        elif gate_condition == "evidence_required":
            # Check if evidence exists for this branch
            has_evidence = _check_branch_evidence(branch, silver_labels, critical_signals)
            branch_result["gated"] = not has_evidence
            if has_evidence:
                branch_result["gate_reason"] = "Evidence found — branch will run"
            else:
                branch_result["gate_reason"] = _build_gate_reason(branch, silver_labels)

        else:
            branch_result["gated"] = False
            branch_result["gate_reason"] = f"Unknown gate condition: {gate_condition}"

        results.append(branch_result)
        status = "GATED OUT" if branch_result["gated"] else "ACTIVE"
        log.info(
            f"Branch {branch['id']} ({branch['type']}): {status} — "
            f"{branch_result['gate_reason']}"
        )

    return results


def _check_branch_evidence(
    branch: dict,
    silver_labels: dict,
    critical_signals: dict,
) -> bool:
    """Check if sufficient evidence exists to run a branch."""
    branch_type = branch.get("type", "")

    if branch_type == "substance_escalation":
        return _has_substance_evidence(silver_labels)

    # Default: evidence exists
    return True


def _has_substance_evidence(silver_labels: dict) -> bool:
    """Check if there's any substance use evidence in the silver labels."""
    sub = silver_labels.get("substance_use") or {}
    if not isinstance(sub, dict):
        return False

    # Check active substance use
    for key in ("alcohol", "drugs"):
        val = sub.get(key, "none")
        if val and val not in ("none", None, ""):
            log.info(f"Substance evidence found: {key} = {val}")
            return True

    # Check positive tox screen
    if sub.get("positive_tox_screen"):
        log.info("Substance evidence found: positive tox screen")
        return True

    # Check detected substances
    detected = sub.get("substances_detected") or []
    if detected:
        log.info(f"Substance evidence found: detected substances = {detected}")
        return True

    log.info("No substance use evidence found in silver labels.")
    return False


def _build_gate_reason(branch: dict, silver_labels: dict) -> str:
    """Build a human-readable gate reason for a gated-out branch."""
    branch_type = branch.get("type", "")

    if branch_type == "substance_escalation":
        sub = silver_labels.get("substance_use") or {}
        parts = []
        for key in ("alcohol", "drugs"):
            val = sub.get(key, "none")
            parts.append(f"{key}={val}")
        tox = sub.get("positive_tox_screen", False)
        parts.append(f"tox_screen={'positive' if tox else 'negative'}")
        return (
            f"NOT APPLICABLE — No substance use evidence. "
            f"Data: {', '.join(parts)}. "
            f"All toxicology screens negative. "
            f"Generating a substance abuse scenario would be clinically unfounded."
        )

    return "NOT APPLICABLE — insufficient evidence"
