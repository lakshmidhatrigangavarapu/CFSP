#!/usr/bin/env python3
"""
Full Pipeline CLI Runner
=========================
Entry point for the full inference pipeline:
  Report Text → Silver Labels → ToT Scenarios → Patient Report

Usage:
  # Single note from JSONL (uses first record, extracts silver labels fresh)
  python -m full_pipeline.run --mode single --input one_sample.jsonl --gpu 0

  # Multi-GPU: shard model across GPUs 1 and 2
  python -m full_pipeline.run --mode single --input one_sample.jsonl --gpu 1,2

  # Single note, reuse existing silver labels from JSONL
  python -m full_pipeline.run --mode single --input one_sample.jsonl --gpu 0 --use-existing-labels

  # Single note from plain text file
  python -m full_pipeline.run --mode single --input my_report.txt --gpu 0

  # Batch processing over JSONL
  python -m full_pipeline.run --mode batch --input one_sample_test10.jsonl --gpu 0

  # Batch with limit
  python -m full_pipeline.run --mode batch --input one_sample_test10.jsonl --gpu 0 --limit 5

  # Custom model / checkpoint
  python -m full_pipeline.run --mode single --input one_sample.jsonl \
      --base-model microsoft/Phi-3.5-mini-instruct \
      --checkpoint ./output/checkpoint-1500 --gpu 0
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from full_pipeline
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline.config import (
    BASE_MODEL_ID,
    CHECKPOINT_PATH,
    NUM_BRANCHES,
    OUTPUT_DIR,
)
from full_pipeline.pipeline import run_single, run_batch


def setup_logging(verbose: bool = False):
    """Configure logging with appropriate format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Full Inference Pipeline: Clinical Note → Silver Labels → "
            "ToT Scenarios → Patient Preparedness Report"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="'single' for one note, 'batch' for JSONL file processing",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file: .jsonl (structured) or .txt (raw note)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU index(es) to use. Single: '0'. Multi-GPU: '1,2' or '0,1,2,3' (default: 0)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL_ID,
        help=f"HuggingFace model ID (default: {BASE_MODEL_ID})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help=f"LoRA adapter checkpoint path (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--num-branches",
        type=int,
        default=NUM_BRANCHES,
        choices=[2, 3],
        help="Number of ToT scenario branches (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(batch mode) Max number of records to process",
    )
    parser.add_argument(
        "--use-existing-labels",
        action="store_true",
        default=False,
        help="Use silver_label from JSONL records instead of re-extracting",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        default=False,
        help="Use base model WITHOUT LoRA adapters (skip finetuned weights). "
             "Useful for testing or when finetuned model produces bad output.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug-level logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("full_pipeline")
    log.info("=" * 60)
    log.info("  Full Inference Pipeline")
    log.info(f"  Mode: {args.mode}")
    log.info(f"  Input: {args.input}")
    log.info(f"  GPU(s): {args.gpu}")
    log.info(f"  Model: {args.base_model}")
    log.info(f"  Checkpoint: {args.checkpoint}")
    log.info(f"  Branches: {args.num_branches}")
    log.info(f"  Base-only (no LoRA): {args.base_only}")
    log.info("=" * 60)

    if args.mode == "single":
        run_single(
            input_source=args.input,
            output_dir=args.output_dir,
            gpu_ids=args.gpu,
            base_model_id=args.base_model,
            checkpoint_path=args.checkpoint,
            num_branches=args.num_branches,
            use_existing_labels=args.use_existing_labels,
            base_only=args.base_only,
        )
    elif args.mode == "batch":
        run_batch(
            input_file=args.input,
            output_dir=args.output_dir,
            gpu_ids=args.gpu,
            base_model_id=args.base_model,
            checkpoint_path=args.checkpoint,
            num_branches=args.num_branches,
            limit=args.limit,
            use_existing_labels=args.use_existing_labels,
            base_only=args.base_only,
        )


if __name__ == "__main__":
    main()
