# Full Pipeline V4: Clinical Report → Silver Labels → Context Enrichment → Gated Scenarios → Report → XAI
# Improvements over V3:
#   - All V3 features preserved
#   - Phase 4: Counterfactual Evidence Attribution (XAI)
#     → Evidence extractor: maps scenario triggers/risk factors to input note spans
#     → Span locator: fuzzy-matches evidence quotes to exact character positions
#     → Branch-colored highlighting: Red (A), Orange (B), Blue (C), Yellow (multi-branch)
#   - DOCX report generator: color-highlighted clinical note + full report document
