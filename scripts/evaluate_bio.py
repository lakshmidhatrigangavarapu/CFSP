#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Clinical Factor Extraction Model
==================================================================

Evaluates the fine-tuned model on the test set, generates metrics,
and logs everything to ClearML under the CFSP-DHATRI project.
"""

import os
import sys
import json
import math
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from clearml import Task, Logger

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.9

CLEARML_PROJECT = "CFSP-DHATRI"
CLEARML_TASK_NAME = "BioMistral-7B-Evaluation"

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def load_model(model_path: str, base_model: str = "BioMistral/BioMistral-7B"):
    """Load fine-tuned model."""
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    model.eval()
    return model, tokenizer


def format_prompt(note: str, schema: Dict) -> str:
    system_prompt = """You are a clinical AI assistant specialized in mental health. 
Extract structured clinical factors from the patient note according to the schema.
Output valid JSON only."""

    schema_str = json.dumps(schema, indent=2)

    return f"""<s>[INST] {system_prompt}

### Clinical Note:
{note}

### Extraction Schema:
{schema_str}

### Instructions:
Extract all relevant clinical factors from the note. Return a JSON object matching the schema.
[/INST]"""


def generate_extraction(model, tokenizer, note: str, schema: Dict,
                        max_note_chars: int = 6000) -> Dict:
    if len(note) > max_note_chars:
        note = note[:max_note_chars] + "... [truncated]"

    prompt = format_prompt(note, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        extracted = json.loads(response.strip())
    except json.JSONDecodeError:
        extracted = {"parse_error": True, "raw_response": response}

    return extracted


def compute_metrics(predictions: List[Dict], references: List[Dict]) -> Dict:
    metrics = {
        "total_examples": len(predictions),
        "parse_errors": 0,
        "exact_match": 0,
        "field_metrics": defaultdict(lambda: {"correct": 0, "total": 0, "present": 0})
    }

    for pred, ref in zip(predictions, references):
        if pred.get("parse_error"):
            metrics["parse_errors"] += 1
            continue

        if pred == ref:
            metrics["exact_match"] += 1

        for field in ref.keys():
            metrics["field_metrics"][field]["total"] += 1
            if field in pred:
                metrics["field_metrics"][field]["present"] += 1
                if pred[field] == ref[field]:
                    metrics["field_metrics"][field]["correct"] += 1

    valid = metrics["total_examples"] - metrics["parse_errors"]
    metrics["parse_error_rate"] = metrics["parse_errors"] / max(metrics["total_examples"], 1)
    metrics["exact_match_rate"] = metrics["exact_match"] / valid if valid > 0 else 0

    # Field-level accuracy & recall (field present in output)
    field_accuracy = {}
    field_recall = {}
    for field, counts in metrics["field_metrics"].items():
        if counts["total"] > 0:
            field_accuracy[field] = counts["correct"] / counts["total"]
            field_recall[field] = counts["present"] / counts["total"]
    metrics["field_accuracy"] = field_accuracy
    metrics["field_recall"] = field_recall
    metrics["avg_field_accuracy"] = (
        sum(field_accuracy.values()) / len(field_accuracy) if field_accuracy else 0)
    metrics["avg_field_recall"] = (
        sum(field_recall.values()) / len(field_recall) if field_recall else 0)

    return metrics


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def evaluate(args):
    logger.info("=" * 60)
    logger.info("Clinical Factor Extraction - Evaluation")
    logger.info("=" * 60)

    # ---- ClearML task ----
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=CLEARML_TASK_NAME,
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
    )
    cl = task.get_logger()

    task.connect({
        "model_path": args.model_path,
        "test_data": args.test_data,
        "max_samples": args.max_samples or "all",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }, name="eval_params")

    # ---- Load model ----
    model, tokenizer = load_model(args.model_path)

    # ---- Schema ----
    schema_path = Path(args.test_data).parent / "extraction_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    task.upload_artifact("extraction_schema", artifact_object=str(schema_path))

    # ---- Test data ----
    logger.info(f"Loading test data: {args.test_data}")
    test_ds = load_dataset("json", data_files=args.test_data, split="train")
    logger.info(f"Test examples: {len(test_ds):,}")

    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
        logger.info(f"Limited to {len(test_ds)} samples")

    task.connect({"test_samples": len(test_ds)}, name="dataset_info")

    # ---- Generate predictions ----
    predictions = []
    references = []

    logger.info("Generating predictions...")
    for idx, example in enumerate(tqdm(test_ds, desc="Evaluating")):
        pred = generate_extraction(model, tokenizer, example["input_text"], schema)
        predictions.append(pred)
        references.append(example["silver_label"])

        # Log progress scalars every 50 samples
        if (idx + 1) % 50 == 0:
            running_parse_err = sum(1 for p in predictions if p.get("parse_error")) / len(predictions)
            cl.report_scalar("Progress/ParseErrorRate", "running",
                             running_parse_err, idx + 1)

    # ---- Compute metrics ----
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, references)

    # ---- Log all metrics to ClearML ----
    cl.report_scalar("Results/ExactMatch", "exact_match_rate",
                     metrics["exact_match_rate"], 0)
    cl.report_scalar("Results/ParseErrorRate", "parse_error_rate",
                     metrics["parse_error_rate"], 0)
    cl.report_scalar("Results/AvgFieldAccuracy", "avg_field_accuracy",
                     metrics["avg_field_accuracy"], 0)
    cl.report_scalar("Results/AvgFieldRecall", "avg_field_recall",
                     metrics["avg_field_recall"], 0)

    # Per-field accuracy as bar chart / table
    for field, acc in metrics["field_accuracy"].items():
        cl.report_scalar("FieldAccuracy", field, acc, 0)
    for field, rec in metrics["field_recall"].items():
        cl.report_scalar("FieldRecall", field, rec, 0)

    # Summary table reported as single-value scalars for the dashboard
    cl.report_single_value("exact_match_rate", metrics["exact_match_rate"])
    cl.report_single_value("parse_error_rate", metrics["parse_error_rate"])
    cl.report_single_value("avg_field_accuracy", metrics["avg_field_accuracy"])
    cl.report_single_value("avg_field_recall", metrics["avg_field_recall"])
    cl.report_single_value("total_examples", metrics["total_examples"])
    cl.report_single_value("parse_errors", metrics["parse_errors"])

    # ---- Save results locally ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v for k, v in metrics.items()
                    if k != "field_metrics"},  # skip defaultdict for JSON
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    preds_path = output_dir / "predictions.jsonl"
    with open(preds_path, "w") as f:
        for pred, ref, example in zip(predictions, references, test_ds):
            f.write(json.dumps({
                "note": example["input_text"][:500] + "...",
                "prediction": pred,
                "reference": ref,
            }) + "\n")

    # Upload artifacts to ClearML
    task.upload_artifact("eval_results", artifact_object=str(results_path))
    task.upload_artifact("predictions", artifact_object=str(preds_path))

    # ---- Console summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Total examples: {metrics['total_examples']}")
    logger.info(f"Parse errors: {metrics['parse_errors']} ({metrics['parse_error_rate']:.1%})")
    logger.info(f"Exact match: {metrics['exact_match_rate']:.1%}")
    logger.info(f"Avg field accuracy: {metrics['avg_field_accuracy']:.1%}")
    logger.info(f"Avg field recall: {metrics['avg_field_recall']:.1%}")
    logger.info("")
    logger.info("Field-level accuracy:")
    for field, acc in sorted(metrics["field_accuracy"].items()):
        logger.info(f"  {field}: {acc:.1%}")
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")

    task.close()
    return metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate clinical factor extraction model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model (best_model or latest_model)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for quick evaluation")

    args = parser.parse_args()
    evaluate(args)
