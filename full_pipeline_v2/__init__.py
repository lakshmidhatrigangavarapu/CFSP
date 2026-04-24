# Full Pipeline V2: Clinical Report → Silver Labels → Validation → Gated Scenarios → Report
# Improvements over V1:
#   - Post-extraction normalizer with ICD code validation
#   - Critical signal override (suicide/violence detection)
#   - Conditional branch gating (skip irrelevant scenarios)
#   - Evidence binding in scenario generation
#   - Final consistency checker
#   - Extended schema (disease_category, progression, cognitive_status)
