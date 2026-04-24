"""
Phase 4: Counterfactual Evidence Attribution (XAI)
====================================================
Maps each generated scenario trigger/risk factor back to specific evidence
spans in the original clinical note. This is the XAI layer:

  - Every scenario gets an EVIDENCE TRAIL back to the patient's own data
  - Each evidence span is tagged with the branch(es) it supports
  - Spans supporting multiple branches are "nexus factors" (highest priority)
  - Ungrounded scenarios (no matching evidence) are flagged

Output: A list of EvidenceSpan objects, each with:
  - text: the matched substring from the clinical note
  - start/end: character positions in the original note
  - branches: list of branch IDs this span supports (A, B, C)
  - factor: the risk factor or trigger this span grounds
  - relevance: high/medium/low
  - is_nexus: True if it supports 2+ branches

This module does NOT use the LLM. It's a post-hoc matching step that runs
after Phase 2 (scenario generation) completes.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy matching evidence quotes to note text
FUZZY_MATCH_THRESHOLD = 0.45

# Maximum characters to search in a sliding window
MAX_WINDOW_SIZE = 300

# Branch color mapping (used downstream by DOCX generator)
BRANCH_COLORS = {
    "A": {"name": "Psychiatric Deterioration", "color": "FF4444", "label": "red"},
    "B": {"name": "Substance Escalation", "color": "FF8C00", "label": "orange"},
    "C": {"name": "Social/Environmental Collapse", "color": "4488FF", "label": "blue"},
    "nexus": {"name": "Multi-Branch (Nexus Factor)", "color": "FFD700", "label": "yellow"},
}


# ============================================================================
# EVIDENCE SPAN DATA STRUCTURE
# ============================================================================


class EvidenceSpan:
    """A span of text in the clinical note that grounds a scenario factor."""

    __slots__ = (
        "text", "start", "end", "branches", "factors",
        "relevance", "is_nexus", "match_score",
    )

    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        branch: str,
        factor: str,
        relevance: str = "high",
        match_score: float = 1.0,
    ):
        self.text = text
        self.start = start
        self.end = end
        self.branches = [branch]
        self.factors = [factor]
        self.relevance = relevance
        self.is_nexus = False
        self.match_score = match_score

    def add_branch(self, branch: str, factor: str):
        """Mark this span as also supporting another branch."""
        if branch not in self.branches:
            self.branches.append(branch)
            self.is_nexus = len(self.branches) > 1
        if factor not in self.factors:
            self.factors.append(factor)

    @property
    def color_key(self) -> str:
        """Return the color key for this span."""
        if self.is_nexus:
            return "nexus"
        return self.branches[0] if self.branches else "A"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "branches": self.branches,
            "factors": self.factors,
            "relevance": self.relevance,
            "is_nexus": self.is_nexus,
            "color_key": self.color_key,
            "match_score": round(self.match_score, 3),
        }


# ============================================================================
# SPAN LOCATOR — fuzzy matching of evidence to note positions
# ============================================================================


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[redacted\]', '', text)
    return text.strip()


def _find_exact_span(note_text: str, query: str) -> Optional[Tuple[int, int]]:
    """Try exact substring match first (case-insensitive)."""
    note_lower = note_text.lower()
    query_lower = query.lower().strip()

    if not query_lower or len(query_lower) < 5:
        return None

    idx = note_lower.find(query_lower)
    if idx >= 0:
        return (idx, idx + len(query_lower))
    return None


def _find_fuzzy_span(
    note_text: str,
    query: str,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> Optional[Tuple[int, int, float]]:
    """
    Fuzzy-match a query string to the best matching span in the note.
    Uses a sliding window with SequenceMatcher.

    Returns (start, end, score) or None.
    """
    note_norm = _normalize_for_matching(note_text)
    query_norm = _normalize_for_matching(query)

    if not query_norm or len(query_norm) < 8:
        return None

    # Determine window size based on query length
    window = min(max(len(query_norm) * 2, 60), MAX_WINDOW_SIZE)
    step = max(len(query_norm) // 4, 10)

    best_score = 0.0
    best_start = 0
    best_end = 0

    for i in range(0, len(note_norm) - min(len(query_norm) // 2, 20), step):
        chunk = note_norm[i:i + window]
        score = SequenceMatcher(None, query_norm, chunk).ratio()

        if score > best_score:
            best_score = score
            best_start = i
            best_end = min(i + window, len(note_norm))

    if best_score < threshold:
        return None

    # Refine: narrow down to the best matching substring
    refined_start, refined_end = _refine_match_boundaries(
        note_norm, query_norm, best_start, best_end
    )

    # Map back to original note positions (approximate — normalization may shift)
    # Use a simple approach: find the corresponding region in original text
    orig_start = _map_normalized_pos_to_original(note_text, refined_start)
    orig_end = _map_normalized_pos_to_original(note_text, refined_end)

    return (orig_start, orig_end, best_score)


def _refine_match_boundaries(
    note_norm: str, query_norm: str, start: int, end: int
) -> Tuple[int, int]:
    """Refine a fuzzy match to tighter boundaries around the actual match."""
    region = note_norm[start:end]

    # Try progressively smaller windows centered on best area
    best_score = 0
    best_s = 0
    best_e = len(region)
    q_len = len(query_norm)

    for s in range(0, max(1, len(region) - q_len // 2), 5):
        for e_offset in range(q_len, min(q_len * 2, len(region) - s), 10):
            sub = region[s:s + e_offset]
            score = SequenceMatcher(None, query_norm, sub).ratio()
            if score > best_score:
                best_score = score
                best_s = s
                best_e = s + e_offset

    return (start + best_s, start + best_e)


def _map_normalized_pos_to_original(original: str, norm_pos: int) -> int:
    """Map a position in normalized text back to approximate original position."""
    # Simple approach: walk through original, count non-extra chars
    norm_idx = 0
    for orig_idx, ch in enumerate(original):
        if norm_idx >= norm_pos:
            return orig_idx
        # In normalization we lowered and collapsed whitespace
        if ch in (' ', '\t', '\n', '\r'):
            # Whitespace collapses — only count one
            if orig_idx > 0 and original[orig_idx - 1] in (' ', '\t', '\n', '\r'):
                continue
        norm_idx += 1
    return len(original)


# ============================================================================
# EVIDENCE EXTRACTION FROM SCENARIOS
# ============================================================================


def _extract_evidence_quotes(scenario_branch: dict) -> List[dict]:
    """
    Extract all evidence quotes and factor descriptions from a single
    scenario branch's output (triggers + scenario narrative data).

    Returns list of {quote, factor, relevance, source}.
    """
    evidence_items = []
    branch_id = scenario_branch.get("scenario_id", "?")

    if scenario_branch.get("gated"):
        return evidence_items

    # Source 1: Triggering factors (Step 1 output)
    triggers = scenario_branch.get("triggering_factors", [])
    if isinstance(triggers, list):
        for t in triggers:
            if not isinstance(t, dict):
                continue
            evidence = t.get("evidence", "")
            factor = t.get("factor", "")
            reason = t.get("reason", "")
            relevance = t.get("mh_relevance", "medium")

            # Use evidence first, fall back to reason, then factor
            quote = evidence or reason or factor
            if quote and len(quote) > 5:
                evidence_items.append({
                    "quote": quote,
                    "factor": factor,
                    "relevance": relevance,
                    "source": "trigger",
                })

    # Source 2: Warning signs from scenario
    scenario_obj = scenario_branch.get("scenario", {})
    if isinstance(scenario_obj, dict):
        # Warning signs often contain note-grounded phrases
        for ws in scenario_obj.get("warning_signs", []):
            if ws and len(ws) > 10:
                evidence_items.append({
                    "quote": ws,
                    "factor": f"warning_sign",
                    "relevance": "medium",
                    "source": "warning_sign",
                })

    # Source 3: Causal chain events may reference specific patient data
    chain = scenario_branch.get("causal_chain", [])
    if isinstance(chain, list):
        for step in chain:
            if not isinstance(step, dict):
                continue
            event = step.get("event", "")
            if event and len(event) > 15:
                evidence_items.append({
                    "quote": event,
                    "factor": f"causal_step_{step.get('step', '?')}",
                    "relevance": "low",
                    "source": "causal_chain",
                })

    return evidence_items


def _extract_from_critical_signals(critical_signals: dict) -> List[dict]:
    """Extract evidence from the critical signal scanner results."""
    evidence_items = []
    if not critical_signals or not critical_signals.get("signals_found"):
        return evidence_items

    categories = critical_signals.get("categories", {})
    for category, matches in categories.items():
        for m in matches:
            matched_text = m.get("matched_text", "")
            context = m.get("context", "")
            if matched_text:
                evidence_items.append({
                    "quote": matched_text,
                    "factor": f"critical_signal_{category}",
                    "relevance": "high",
                    "source": "critical_signal",
                    "branches": ["A"],  # Critical signals primarily ground Branch A
                })
    return evidence_items


def _extract_from_enriched_context(silver_labels: dict) -> List[dict]:
    """Extract evidence from the context enricher results."""
    evidence_items = []
    ctx = silver_labels.get("enriched_context") or {}

    branch_map = {
        "housing_crisis": "C",
        "caregiver_loss": "C",
        "trauma_history": "A",
        "weight_change": "A",
        "functional_decline": "A",
        "clinician_prognosis": "A",
        "legal_status": "A",
    }

    for key, branch in branch_map.items():
        entry = ctx.get(key, {})
        if entry and entry.get("detected"):
            evidence_text = entry.get("evidence", "")
            details = entry.get("details", [])
            if evidence_text and len(evidence_text) > 10:
                evidence_items.append({
                    "quote": evidence_text,
                    "factor": key,
                    "relevance": "high",
                    "source": "enriched_context",
                    "branches": [branch],
                })
            # Also try individual detail labels as search terms
            for detail in details:
                if detail and len(detail) > 5:
                    evidence_items.append({
                        "quote": detail,
                        "factor": key,
                        "relevance": "medium",
                        "source": "enriched_context_detail",
                        "branches": [branch],
                    })

    return evidence_items


# ============================================================================
# SPAN MERGING — combine overlapping spans and detect nexus factors
# ============================================================================


def _merge_overlapping_spans(spans: List[EvidenceSpan]) -> List[EvidenceSpan]:
    """
    Merge overlapping spans. If two spans from different branches overlap,
    the merged span becomes a nexus factor (yellow highlight).
    """
    if not spans:
        return spans

    # Sort by start position
    spans.sort(key=lambda s: s.start)

    merged = []
    current = spans[0]

    for next_span in spans[1:]:
        # Check if overlapping (or adjacent within 5 chars)
        if next_span.start <= current.end + 5:
            # Merge: extend end, combine branches
            current.end = max(current.end, next_span.end)
            current.text = ""  # Will be re-extracted from note later
            for b in next_span.branches:
                current.add_branch(b, next_span.factors[0] if next_span.factors else "")
            for f in next_span.factors:
                if f not in current.factors:
                    current.factors.append(f)
            current.match_score = max(current.match_score, next_span.match_score)
        else:
            merged.append(current)
            current = next_span

    merged.append(current)
    return merged


# ============================================================================
# MAIN XAI FUNCTION
# ============================================================================


def extract_evidence_attribution(
    note_text: str,
    scenarios_result: dict,
    silver_labels: dict,
    critical_signals: dict = None,
) -> dict:
    """
    Phase 4: Counterfactual Evidence Attribution.

    Maps scenario outputs back to specific evidence spans in the original
    clinical note. Each span is tagged with branch color(s).

    Args:
        note_text: Original clinical note text (cleaned).
        scenarios_result: Phase 2 output (scenarios with triggers/chains).
        silver_labels: Phase 1 output (extracted factors).
        critical_signals: Phase 1.6 output (critical signal matches).

    Returns:
        Dict containing:
          - evidence_spans: list of span dicts (sorted by position)
          - branch_summary: per-branch evidence count
          - nexus_factors: list of multi-branch spans
          - ungrounded_branches: branches with no evidence in note
          - coverage_stats: how much of the note is evidence-backed
    """
    log.info("Phase 4: Running Counterfactual Evidence Attribution (XAI) ...")

    all_spans: List[EvidenceSpan] = []
    branch_evidence_count = {"A": 0, "B": 0, "C": 0}

    # --- Collect evidence from all sources ---

    # 1. Scenario triggers and chains (primary evidence)
    for scenario in scenarios_result.get("scenarios", []):
        branch_id = scenario.get("scenario_id", "?")
        if scenario.get("gated"):
            continue

        evidence_items = _extract_evidence_quotes(scenario)
        for item in evidence_items:
            span = _locate_evidence_in_note(
                note_text, item["quote"], branch_id,
                item["factor"], item["relevance"]
            )
            if span:
                all_spans.append(span)
                branch_evidence_count[branch_id] = branch_evidence_count.get(branch_id, 0) + 1

    # 2. Enriched context evidence (pre-located by context_enricher)
    context_evidence = _extract_from_enriched_context(silver_labels)
    for item in context_evidence:
        branches = item.get("branches", ["A"])
        for branch_id in branches:
            span = _locate_evidence_in_note(
                note_text, item["quote"], branch_id,
                item["factor"], item["relevance"]
            )
            if span:
                all_spans.append(span)
                branch_evidence_count[branch_id] = branch_evidence_count.get(branch_id, 0) + 1

    # 3. Critical signals (already located by signal scanner)
    if critical_signals:
        signal_evidence = _extract_from_critical_signals(critical_signals)
        for item in signal_evidence:
            branches = item.get("branches", ["A"])
            for branch_id in branches:
                span = _locate_evidence_in_note(
                    note_text, item["quote"], branch_id,
                    item["factor"], item["relevance"]
                )
                if span:
                    all_spans.append(span)
                    branch_evidence_count[branch_id] = branch_evidence_count.get(branch_id, 0) + 1

    # --- Merge overlapping spans and detect nexus factors ---
    merged_spans = _merge_overlapping_spans(all_spans)

    # Re-extract text for merged spans that lost their text
    for span in merged_spans:
        if not span.text and span.start < span.end <= len(note_text):
            span.text = note_text[span.start:span.end]

    # --- Build output ---
    nexus_factors = [s for s in merged_spans if s.is_nexus]
    ungrounded = [
        branch_id for branch_id, count in branch_evidence_count.items()
        if count == 0
    ]

    # Filter out gated branches from ungrounded list
    gated_ids = {
        s.get("scenario_id") for s in scenarios_result.get("scenarios", [])
        if s.get("gated")
    }
    ungrounded = [b for b in ungrounded if b not in gated_ids]

    # Coverage: what fraction of the note is highlighted
    highlighted_chars = sum(s.end - s.start for s in merged_spans)
    total_chars = len(note_text)
    coverage = highlighted_chars / total_chars if total_chars > 0 else 0

    result = {
        "evidence_spans": [s.to_dict() for s in merged_spans],
        "branch_summary": {
            bid: {
                "evidence_count": count,
                "color": BRANCH_COLORS.get(bid, {}).get("color", "CCCCCC"),
                "label": BRANCH_COLORS.get(bid, {}).get("label", "gray"),
            }
            for bid, count in branch_evidence_count.items()
        },
        "nexus_factors": [s.to_dict() for s in nexus_factors],
        "nexus_count": len(nexus_factors),
        "ungrounded_branches": ungrounded,
        "coverage_stats": {
            "highlighted_chars": highlighted_chars,
            "total_chars": total_chars,
            "coverage_pct": round(coverage * 100, 1),
        },
        "total_spans": len(merged_spans),
    }

    log.info(
        f"Phase 4: Found {len(merged_spans)} evidence spans "
        f"({len(nexus_factors)} nexus factors, "
        f"{coverage:.1%} note coverage)"
    )
    if ungrounded:
        log.warning(f"Phase 4: Ungrounded branches (no evidence): {ungrounded}")

    return result


def _locate_evidence_in_note(
    note_text: str,
    query: str,
    branch_id: str,
    factor: str,
    relevance: str,
) -> Optional[EvidenceSpan]:
    """
    Try to locate a piece of evidence in the clinical note.
    Uses exact match first, then fuzzy matching.
    """
    if not query or len(query) < 5:
        return None

    # Clean up query — remove common prefixes from model output
    query = re.sub(r'^(?:Patient|The patient|Data shows|Evidence:)\s*', '', query, flags=re.IGNORECASE)
    query = query.strip('"\'')

    # Try exact match
    exact = _find_exact_span(note_text, query)
    if exact:
        start, end = exact
        return EvidenceSpan(
            text=note_text[start:end],
            start=start, end=end,
            branch=branch_id,
            factor=factor,
            relevance=relevance,
            match_score=1.0,
        )

    # Try fuzzy match
    fuzzy = _find_fuzzy_span(note_text, query)
    if fuzzy:
        start, end, score = fuzzy
        matched_text = note_text[start:end]
        # Don't accept very long fuzzy matches (likely capturing too much context)
        if len(matched_text) < 500:
            return EvidenceSpan(
                text=matched_text,
                start=start, end=end,
                branch=branch_id,
                factor=factor,
                relevance=relevance,
                match_score=score,
            )

    log.debug(f"Could not locate evidence in note: '{query[:80]}...'")
    return None
