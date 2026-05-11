#!/usr/bin/env python3
"""
Generate PDF report from existing JSON result file.
Usage:
    python -m full_pipeline_v4.generate_pdf_from_json <json_file> [--output <pdf_path>]
    
Example:
    python -m full_pipeline_v4.generate_pdf_from_json output/result_v4_ui_ui.json
"""

import argparse
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
log = logging.getLogger("generate_pdf")

# Add package to path
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline_v4.pdf_generator import generate_pdf_report
from full_pipeline_v4.config import OUTPUT_DIR


def load_clinical_note(json_path: Path) -> str:
    """Try to load the original clinical note from various sources."""
    # Try to find the note in the same directory
    result_dir = json_path.parent
    
    # Check for input text file
    note_files = [
        result_dir / "input_note.txt",
        result_dir / f"note_{json_path.stem.replace('result_v4_', '')}.txt",
    ]
    
    for nf in note_files:
        if nf.exists():
            log.info(f"Found clinical note file: {nf}")
            return nf.read_text(encoding="utf-8")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate PDF from JSON result file")
    parser.add_argument("json_file", type=str, help="Path to the JSON result file")
    parser.add_argument("--output", "-o", type=str, help="Output PDF path (optional)")
    parser.add_argument("--note", "-n", type=str, help="Path to clinical note file (optional)")
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
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
    
    subject_id = result.get("subject_id", "unknown")
    hadm_id = result.get("hadm_id", "unknown")
    
    # Get clinical note
    note_text = ""
    if args.note:
        note_path = Path(args.note)
        if note_path.exists():
            note_text = note_path.read_text(encoding="utf-8")
            log.info(f"Loaded clinical note from: {note_path}")
        else:
            log.warning(f"Note file not found: {note_path}")
    
    if not note_text:
        note_text = load_clinical_note(json_path)
    
    if not note_text:
        # Create placeholder note from evidence spans
        log.warning("No clinical note file found. Using evidence spans as placeholder.")
        spans = evidence_result.get("evidence_spans", [])
        note_parts = []
        for span in spans[:10]:
            text = span.get("text", "")
            if text:
                note_parts.append(text)
        note_text = "\n\n".join(note_parts) if note_parts else "Clinical note not available."
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"report_v4_{subject_id}_{hadm_id}_new.pdf")
    
    log.info(f"Generating PDF report...")
    log.info(f"  Subject ID: {subject_id}")
    log.info(f"  Admission ID: {hadm_id}")
    log.info(f"  Risk Tier: {report.get('overall_risk_tier', 'UNKNOWN')}")
    log.info(f"  Evidence Spans: {len(evidence_result.get('evidence_spans', []))}")
    
    # Generate PDF
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
    else:
        log.error("❌ PDF generation failed. Is reportlab installed?")
        sys.exit(1)


if __name__ == "__main__":
    main()
