#!/usr/bin/env python3
"""
Full Pipeline V3 CLI Runner
=============================
Usage:
  python -m full_pipeline_v3 --mode single --input one_sample.jsonl --gpu 0
  python -m full_pipeline_v3 --mode single --input input-output/input.txt --gpu 0
  python -m full_pipeline_v3 --mode batch --input one_sample_test10.jsonl --gpu 0,1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline_v3.config import (
    BASE_MODEL_ID,
    CHECKPOINT_PATH,
    NUM_BRANCHES,
    OUTPUT_DIR,
)
from full_pipeline_v3.pipeline import run_single, run_batch


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Full Inference Pipeline V3: Clinical Note → Silver Labels → "
            "Normalization → Context Enrichment → Signal Scan → Gated Scenarios → Report → Consistency Check"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--num-branches", type=int, default=NUM_BRANCHES, choices=[2, 3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-existing-labels", action="store_true", default=False)
    parser.add_argument("--base-only", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("full_pipeline_v3")
    log.info("=" * 60)
    log.info("  Full Inference Pipeline V3")
    log.info(f"  Mode: {args.mode}")
    log.info(f"  Input: {args.input}")
    log.info(f"  GPU(s): {args.gpu}")
    log.info(f"  Model: {args.base_model}")
    log.info(f"  Checkpoint: {args.checkpoint}")
    log.info(f"  Branches: {args.num_branches}")
    log.info(f"  Base-only (no LoRA): {args.base_only}")
    log.info("  V3 Features: normalization, context enrichment, negation-aware signals, branch gating, consistency check")
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
