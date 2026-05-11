#!/usr/bin/env python3
"""
Quick script to regenerate PDF from the result_v4_ui_ui.json with the new PDF generator.
Run: python full_pipeline_v4/regenerate_ui_ui_pdf.py
"""

import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regenerate_pdf")

# Add package to path
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline_v4.pdf_generator import generate_pdf_report

def main():
    # Load the JSON result
    json_path = SCRIPT_DIR / "output" / "result_v4_ui_ui.json"
    
    if not json_path.exists():
        log.error(f"JSON file not found: {json_path}")
        sys.exit(1)
    
    log.info(f"Loading result from: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    # Extract components
    silver_labels = result.get("phase1_silver_labels", {})
    scenarios_result = result.get("phase2_scenarios", {})
    report = result.get("phase3_report", {})
    evidence_result = result.get("phase4_evidence_attribution", {})
    critical_signals = result.get("phase1_6_critical_signals", {})
    
    subject_id = result.get("subject_id", "ui")
    hadm_id = result.get("hadm_id", "ui")
    
    # Reconstruct clinical note from evidence spans
    # The evidence spans contain portions of the original note with positions
    spans = evidence_result.get("evidence_spans", [])
    
    # Sort by position and create a placeholder note
    sorted_spans = sorted(spans, key=lambda s: s.get("start", 0))
    
    # Build a synthetic note from the spans (for highlighting purposes)
    # We'll create padding text to maintain positions
    max_end = max((s.get("end", 0) for s in spans), default=15000)
    note_text = " " * max_end  # Create placeholder
    
    # Fill in the actual text at the correct positions
    note_chars = list(note_text)
    for span in sorted_spans:
        text = span.get("text", "")
        start = span.get("start", 0)
        end = span.get("end", start + len(text))
        
        # Ensure we don't go out of bounds
        if start < len(note_chars):
            for i, char in enumerate(text):
                if start + i < len(note_chars):
                    note_chars[start + i] = char
    
    note_text = "".join(note_chars)
    
    # Clean up excessive whitespace for better display
    # But preserve the general structure for highlighting
    log.info(f"Reconstructed note length: {len(note_text)} chars")
    log.info(f"Evidence spans: {len(spans)}")
    log.info(f"Risk tier: {report.get('overall_risk_tier', 'UNKNOWN')}")
    
    # Generate PDF with a different name to compare
    output_path = str(SCRIPT_DIR / "output" / f"report_v4_{subject_id}_{hadm_id}_NEW_FORMAT.pdf")
    
    log.info(f"Generating PDF report with new format...")
    
    pdf_path = generate_pdf_report(
        note_text=note_text,
        evidence_result=evidence_result,
        report=report,
        silver_labels=silver_labels,
        scenarios_result=scenarios_result,
        critical_signals=critical_signals,
        output_path=output_path,
        subject_id=str(subject_id),
        hadm_id=str(hadm_id),
    )
    
    if pdf_path:
        log.info(f"✅ PDF report generated: {pdf_path}")
        print(f"\n{'='*60}")
        print(f"SUCCESS! New PDF generated:")
        print(f"  {pdf_path}")
        print(f"{'='*60}")
    else:
        log.error("❌ PDF generation failed. Is reportlab installed?")
        print("Run: pip install reportlab")
        sys.exit(1)


if __name__ == "__main__":
    main()
