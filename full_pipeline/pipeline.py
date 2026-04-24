"""
Full Inference Pipeline: Orchestrator
======================================
Unified pipeline that chains all three phases:

  Phase 1: Silver Label Extraction (finetuned model)
           Clinical note → structured clinical factors (JSON)

  Phase 2: Tree-of-Thoughts Scenario Generation
           Silver labels + note → 3 counterfactual negative trajectories

  Phase 3: Patient Preparedness Report
           Silver labels + scenarios → concise clinical briefing

The pipeline loads the model ONCE and runs all phases sequentially.
Supports single-note and batch (JSONL) modes.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import BASE_MODEL_ID, CHECKPOINT_PATH, NUM_BRANCHES, OUTPUT_DIR
from .model_loader import load_model
from .silver_label_extractor import extract_silver_labels
from .scenario_generator import generate_scenarios
from .report_generator import generate_report, format_report_text
from .utils import clean_report_text

log = logging.getLogger(__name__)


# ============================================================================
# SINGLE NOTE PIPELINE
# ============================================================================


def run_pipeline(
    report_text: str,
    model,
    tokenizer,
    device: str,
    num_branches: int = NUM_BRANCHES,
    silver_labels: Optional[dict] = None,
    use_finetuned: bool = True,
) -> dict:
    """
    Run the full 3-phase pipeline on a single clinical note.

    Args:
        report_text: Raw clinical note text.
        model: Loaded model (with LoRA adapters).
        tokenizer: Loaded tokenizer.
        device: CUDA device string.
        num_branches: Number of ToT scenario branches (2 or 3).
        silver_labels: Pre-extracted silver labels. If None, Phase 1
                       extracts them from the report text.
        use_finetuned: If True, use finetuned prompt format for extraction.
                       If False, use base-model chat template.

    Returns:
        Dictionary containing all pipeline outputs.
    """
    pipeline_start = datetime.now()
    cleaned_text = clean_report_text(report_text)

    # ================================================================
    # PHASE 1: Silver Label Extraction
    # ================================================================
    if silver_labels is None:
        silver_labels = extract_silver_labels(
            report_text=cleaned_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_finetuned=use_finetuned,
        )
    else:
        log.info("Phase 1: Using pre-provided silver labels (skipping extraction).")

    phase1_ok = not silver_labels.get("_parse_failed", False)
    if not phase1_ok:
        log.warning("Phase 1 extraction was partial — continuing with available data.")
        # Remove the failure flag so downstream phases can still use whatever was extracted
        silver_labels.pop("_parse_failed", None)
        silver_labels.pop("_raw_text", None)

    # ================================================================
    # PHASE 2: Tree-of-Thoughts Scenario Generation
    # ================================================================
    # Note: LoRA weights are merged into the base model via merge_and_unload(),
    # so there's no separate adapter to toggle. The merged model handles all phases.
    scenarios_result = generate_scenarios(
        report_text=cleaned_text,
        silver_labels=silver_labels,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_branches=num_branches,
    )

    # ================================================================
    # PHASE 3: Patient Preparedness Report
    # ================================================================
    report = generate_report(
        report_text=cleaned_text,
        silver_labels=silver_labels,
        scenarios_result=scenarios_result,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # ================================================================
    # ASSEMBLE FINAL OUTPUT
    # ================================================================
    pipeline_end = datetime.now()
    duration = (pipeline_end - pipeline_start).total_seconds()

    result = {
        "pipeline_version": "1.0",
        "timestamp": pipeline_start.isoformat(),
        "duration_seconds": round(duration, 1),
        "status": "success",
        "phase1_silver_labels": silver_labels,
        "phase2_scenarios": scenarios_result,
        "phase3_report": report,
    }

    log.info(f"Pipeline complete in {duration:.1f}s")
    return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def run_batch(
    input_file: str,
    output_dir: str = OUTPUT_DIR,
    gpu_ids: str = "0",
    base_model_id: str = BASE_MODEL_ID,
    checkpoint_path: str = CHECKPOINT_PATH,
    num_branches: int = NUM_BRANCHES,
    limit: Optional[int] = None,
    use_existing_labels: bool = False,
    base_only: bool = False,
):
    """
    Process a JSONL file of clinical notes through the full pipeline.

    Each line in the input file must have:
      - "input_text": the clinical note
      - (optional) "subject_id", "hadm_id": patient identifiers
      - (optional) "silver_label": pre-extracted labels (skip Phase 1)

    Args:
        input_file: Path to input JSONL file.
        output_dir: Directory for output files.
        gpu_ids: Comma-separated GPU indices (e.g. "0" or "1,2,3").
        base_model_id: HuggingFace model ID.
        checkpoint_path: Path to LoRA checkpoint.
        num_branches: Number of scenario branches.
        limit: Max records to process (None = all).
        use_existing_labels: If True and silver_label exists in the record,
                             skip Phase 1 extraction.
        base_only: If True, skip LoRA adapters and use base model only.
    """
    multi_gpu = "," in gpu_ids
    load_adapters = not base_only

    if multi_gpu:
        # Multi-GPU: set CUDA_VISIBLE_DEVICES so device_map="auto" uses only those GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        gpu_id = 0
        gpu_count_str = f"{len(gpu_ids.split(','))} GPUs ({gpu_ids})"
    else:
        # Single GPU: pass the actual index directly — no CUDA_VISIBLE_DEVICES remapping
        gpu_id = int(gpu_ids)
        gpu_count_str = f"GPU {gpu_id}"

    adapter_str = "with LoRA adapters" if load_adapters else "BASE MODEL ONLY (no adapters)"
    log.info(f"Loading model {adapter_str} on {gpu_count_str} ...")
    model, tokenizer, device = load_model(
        gpu_id=gpu_id,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        load_adapters=load_adapters,
        multi_gpu=multi_gpu,
    )

    # Prepare output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = out_path / f"pipeline_results_{timestamp}.jsonl"
    reports_dir = out_path / f"reports_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_file)
    processed, skipped = 0, 0

    with open(input_path) as fin, open(results_file, "w") as fout:
        for line_num, line in enumerate(fin):
            if limit and processed >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_num}: JSON parse error — {e}")
                skipped += 1
                continue

            subject_id = record.get("subject_id", f"line_{line_num}")
            hadm_id = record.get("hadm_id", "unknown")
            note_text = record.get("input_text", "")

            if not note_text:
                log.warning(f"Record {subject_id}/{hadm_id}: missing note text. Skipping.")
                skipped += 1
                continue

            # Check for pre-existing silver labels
            existing_labels = None
            if use_existing_labels:
                existing_labels = record.get("silver_label") or record.get("extracted_factors")

            log.info(f"\n{'='*60}")
            log.info(f"Processing patient {subject_id} / admission {hadm_id} "
                     f"(record {processed + 1})")
            log.info(f"{'='*60}")

            try:
                result = run_pipeline(
                    report_text=note_text,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    num_branches=num_branches,
                    silver_labels=existing_labels,
                    use_finetuned=not base_only,
                )

                result["subject_id"] = subject_id
                result["hadm_id"] = hadm_id

                # Write JSONL result
                fout.write(json.dumps(result) + "\n")
                fout.flush()

                # Write human-readable report
                if result.get("status") == "success":
                    report_text_formatted = format_report_text(
                        report=result["phase3_report"],
                        silver_labels=result["phase1_silver_labels"],
                        scenarios_result=result["phase2_scenarios"],
                    )
                    report_file = reports_dir / f"report_{subject_id}_{hadm_id}.txt"
                    with open(report_file, "w") as rf:
                        rf.write(report_text_formatted)

                processed += 1

            except Exception as e:
                log.error(f"Record {subject_id}/{hadm_id}: pipeline failed — {e}")
                import traceback
                log.error(traceback.format_exc())
                skipped += 1

    log.info(f"\nBatch complete. Processed: {processed} | Skipped: {skipped}")
    log.info(f"Results: {results_file}")
    log.info(f"Reports: {reports_dir}")


# ============================================================================
# SINGLE NOTE DEMO (interactive)
# ============================================================================


def run_single(
    input_source: str,
    output_dir: str = OUTPUT_DIR,
    gpu_ids: str = "0",
    base_model_id: str = BASE_MODEL_ID,
    checkpoint_path: str = CHECKPOINT_PATH,
    num_branches: int = NUM_BRANCHES,
    use_existing_labels: bool = False,
    base_only: bool = False,
):
    """
    Run the full pipeline on a single note (from file or JSONL first record).

    Args:
        input_source: Path to a .jsonl file (uses first record) or a .txt
                      file with raw clinical note text.
        output_dir: Directory for output.
        gpu_ids: Comma-separated GPU indices (e.g. "0" or "1,2,3").
        base_model_id: HuggingFace model ID.
        checkpoint_path: Path to LoRA checkpoint.
        num_branches: Number of scenario branches.
        use_existing_labels: Use silver_label from JSONL if available.
        base_only: If True, skip LoRA adapters and use base model only.
    """
    multi_gpu = "," in gpu_ids
    load_adapters = not base_only

    if multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        gpu_id = 0
        gpu_count_str = f"{len(gpu_ids.split(','))} GPUs ({gpu_ids})"
    else:
        gpu_id = int(gpu_ids)
        gpu_count_str = f"GPU {gpu_id}"

    adapter_str = "with LoRA adapters" if load_adapters else "BASE MODEL ONLY (no adapters)"
    log.info(f"Loading model {adapter_str} on {gpu_count_str} ...")
    model, tokenizer, device = load_model(
        gpu_id=gpu_id,
        base_model_id=base_model_id,
        checkpoint_path=checkpoint_path,
        load_adapters=load_adapters,
        multi_gpu=multi_gpu,
    )

    # Read input
    input_path = Path(input_source)
    note_text = ""
    existing_labels = None
    subject_id = "demo"
    hadm_id = "demo"

    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            record = json.loads(f.readline())
        note_text = record.get("input_text", "")
        subject_id = record.get("subject_id", "demo")
        hadm_id = record.get("hadm_id", "demo")
        if use_existing_labels:
            existing_labels = record.get("silver_label") or record.get("extracted_factors")
    elif input_path.suffix == ".txt":
        note_text = input_path.read_text()
    else:
        # Try reading as plain text
        note_text = input_path.read_text()

    if not note_text:
        log.error(f"No text found in {input_source}")
        return

    log.info(f"Running pipeline for subject={subject_id}, hadm={hadm_id} ...")

    result = run_pipeline(
        report_text=note_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_branches=num_branches,
        silver_labels=existing_labels,
        use_finetuned=not base_only,
    )
    result["subject_id"] = subject_id
    result["hadm_id"] = hadm_id

    # Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Full JSON result
    json_file = out_path / f"result_{subject_id}_{hadm_id}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Full result saved → {json_file}")

    # Human-readable report
    if result.get("status") == "success":
        report_formatted = format_report_text(
            report=result["phase3_report"],
            silver_labels=result["phase1_silver_labels"],
            scenarios_result=result["phase2_scenarios"],
        )

        report_file = out_path / f"report_{subject_id}_{hadm_id}.txt"
        with open(report_file, "w") as f:
            f.write(report_formatted)
        log.info(f"Report saved → {report_file}")

        # Also print to console
        print(report_formatted)
    else:
        log.error(f"Pipeline failed: {result.get('error', 'unknown')}")
        print(json.dumps(result, indent=2))
