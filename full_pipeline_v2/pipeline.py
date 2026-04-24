"""
Full Inference Pipeline V2: Orchestrator
==========================================
Unified pipeline that chains all phases with V2 improvements:

  Phase 1:   Silver Label Extraction (finetuned model)
  Phase 1.5: Post-Extraction Normalization (ICD validation, Dx reconciliation)
  Phase 1.6: Critical Signal Scanning (rule-based, no model)
  Phase 1.7: Branch Gating (evidence-based conditional logic)
  Phase 2:   Tree-of-Thoughts Scenario Generation (gated, evidence-bound)
  Phase 3:   Patient Preparedness Report
  Phase 3.5: Final Consistency Check (risk tier, contradictions)

Key V2 improvements:
  - Wrong diagnosis labels get caught and corrected
  - Irrelevant branches (e.g., substance use with no evidence) are skipped
  - Suicide/violence signals are NEVER missed (rule-based override)
  - Risk tier contradictions are resolved deterministically
  - Evidence binding prevents hallucinated pathways
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
from .normalizer import normalize_silver_labels
from .critical_signals import scan_critical_signals, augment_silver_labels_with_signals
from .branch_gating import evaluate_branch_gates
from .scenario_generator import generate_scenarios
from .report_generator import generate_report, format_report_text
from .consistency_checker import check_consistency
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
    Run the full V2 pipeline on a single clinical note.

    Pipeline flow:
      1.  Extract silver labels (or use pre-provided)
      1.5 Normalize: ICD validation, Dx reconciliation, disease classification
      1.6 Scan critical signals: suicide/violence detection (rule-based)
      1.7 Evaluate branch gates: skip irrelevant scenario branches
      2.  Generate scenarios (only for active branches)
      3.  Generate report
      3.5 Final consistency check

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
        silver_labels.pop("_parse_failed", None)
        silver_labels.pop("_raw_text", None)

    # ================================================================
    # PHASE 1.5: Post-Extraction Normalization (NEW in V2)
    # ================================================================
    log.info("Phase 1.5: Running post-extraction normalization ...")
    silver_labels = normalize_silver_labels(
        silver_labels=silver_labels,
        note_text=cleaned_text,
    )

    # ================================================================
    # PHASE 1.6: Critical Signal Scanning (NEW in V2)
    # ================================================================
    log.info("Phase 1.6: Scanning for critical safety signals ...")
    critical_signals = scan_critical_signals(cleaned_text)

    # Augment silver labels with any signals the model missed
    silver_labels = augment_silver_labels_with_signals(
        silver_labels=silver_labels,
        critical_signals=critical_signals,
        note_text=cleaned_text,
    )

    # ================================================================
    # PHASE 1.7: Branch Gating (NEW in V2)
    # ================================================================
    log.info("Phase 1.7: Evaluating branch gates ...")
    gated_branches = evaluate_branch_gates(
        silver_labels=silver_labels,
        critical_signals=critical_signals,
    )

    # ================================================================
    # PHASE 2: Tree-of-Thoughts Scenario Generation
    # ================================================================
    scenarios_result = generate_scenarios(
        report_text=cleaned_text,
        silver_labels=silver_labels,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_branches=num_branches,
        gated_branches=gated_branches,
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
        critical_signals=critical_signals,
    )

    # ================================================================
    # PHASE 3.5: Final Consistency Check (NEW in V2)
    # ================================================================
    log.info("Phase 3.5: Running final consistency check ...")
    report = check_consistency(
        report=report,
        silver_labels=silver_labels,
        scenarios_result=scenarios_result,
        critical_signals=critical_signals,
    )

    # ================================================================
    # ASSEMBLE FINAL OUTPUT
    # ================================================================
    pipeline_end = datetime.now()
    duration = (pipeline_end - pipeline_start).total_seconds()

    result = {
        "pipeline_version": "2.0",
        "timestamp": pipeline_start.isoformat(),
        "duration_seconds": round(duration, 1),
        "status": "success",
        "phase1_silver_labels": silver_labels,
        "phase1_5_normalization_log": silver_labels.get("_normalization_log", []),
        "phase1_6_critical_signals": critical_signals,
        "phase1_7_branch_gating": [
            {"id": b["id"], "type": b["type"], "gated": b["gated"], "reason": b["gate_reason"]}
            for b in gated_branches
        ],
        "phase2_scenarios": scenarios_result,
        "phase3_report": report,
        "phase3_5_consistency_log": report.get("_consistency_log", []),
    }

    log.info(f"Pipeline V2 complete in {duration:.1f}s")
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
    """Process a JSONL file of clinical notes through the full V2 pipeline."""
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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = out_path / f"pipeline_v2_results_{timestamp}.jsonl"
    reports_dir = out_path / f"reports_v2_{timestamp}"
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

                fout.write(json.dumps(result) + "\n")
                fout.flush()

                if result.get("status") == "success":
                    critical_signals = result.get("phase1_6_critical_signals", {})
                    report_text_formatted = format_report_text(
                        report=result["phase3_report"],
                        silver_labels=result["phase1_silver_labels"],
                        scenarios_result=result["phase2_scenarios"],
                        critical_signals=critical_signals,
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
# SINGLE NOTE
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
    """Run the full V2 pipeline on a single note."""
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
        note_text = input_path.read_text()

    if not note_text:
        log.error(f"No text found in {input_source}")
        return

    log.info(f"Running V2 pipeline for subject={subject_id}, hadm={hadm_id} ...")

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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / f"result_v2_{subject_id}_{hadm_id}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Full result saved → {json_file}")

    if result.get("status") == "success":
        critical_signals = result.get("phase1_6_critical_signals", {})
        report_formatted = format_report_text(
            report=result["phase3_report"],
            silver_labels=result["phase1_silver_labels"],
            scenarios_result=result["phase2_scenarios"],
            critical_signals=critical_signals,
        )

        report_file = out_path / f"report_v2_{subject_id}_{hadm_id}.txt"
        with open(report_file, "w") as f:
            f.write(report_formatted)
        log.info(f"Report saved → {report_file}")

        print(report_formatted)
    else:
        log.error(f"Pipeline failed: {result.get('error', 'unknown')}")
