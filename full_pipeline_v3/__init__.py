# Full Pipeline V3: Clinical Report → Silver Labels → Context Enrichment → Gated Scenarios → Report
# Improvements over V2:
#   - Context enricher (Phase 1.55): housing, trauma, caregiver, weight, functional decline, prognosis
#   - Negation-aware signal scanner (reduces false positive alerts)
#   - Duplicate alert deduplication
#   - ICD code display fix (shows corrected code after normalization)
#   - Severity inference when model returns "unknown"
#   - Hallucinated <|user|> turn truncation
#   - Enriched compact_factors() with context fields for scenario generator
#   - Evidence binding in scenario generation
#   - Final consistency checker
#   - Extended schema (disease_category, progression, cognitive_status)
