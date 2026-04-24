#!/usr/bin/env python3

import os
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# ================= GPU SETUP =================
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6,7"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from clearml import Task

# ================= CONFIG =================

MAX_NEW_TOKENS = 256
BATCH_SIZE = 4

CLEARML_PROJECT = "CFSP-DHATRI"
CLEARML_TASK_NAME = "Phi3.5-mini-Evaluation"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= CLEARML =================

def setup_clearml():
    try:
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=CLEARML_TASK_NAME,
            task_type=Task.TaskTypes.testing,
            reuse_last_task_id=False,
        )
        return task, task.get_logger()
    except Exception:
        logger.warning("ClearML disabled")
        return None, None

# ================= MODEL =================

def load_model(model_path: str, base_model="microsoft/Phi-3.5-mini-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ✅ FIX: LEFT PADDING
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {0: "20GiB", 1: "20GiB", 2: "20GiB"}

    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        logger.info("Loading LoRA model")

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory
        )

        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    else:
        logger.info("Loading full model")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory
        )

    model.eval()
    return model, tokenizer

# ================= PROMPT =================

def format_prompt(note: str, schema: Dict) -> str:
    return f"""<|system|>
You are a clinical information extraction system.

STRICT RULES:
- Output ONLY valid JSON
- NO explanations
- NO extra text
- JSON must start with {{ and end with }}

<|end|>

<|user|>
Extract structured data from this note.

NOTE:
{note}

SCHEMA:
{json.dumps(schema)}

<|end|>

<|assistant|>
"""

# ================= GENERATION =================

def batch_generate(model, tokenizer, notes: List[str], schema: Dict):
    prompts = [format_prompt(n, schema) for n in notes]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,       # ✅ deterministic
            top_p=1.0,
            do_sample=False,       # ✅ no randomness
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    for i, out in enumerate(decoded):
        prompt_len = len(tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True))
        text = out[prompt_len:]

        try:
            text = text.strip()

            # ✅ Extract JSON safely
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group()

            parsed = json.loads(text)
            results.append(parsed)

        except:
            results.append({"parse_error": True, "raw": text})

    return results

# ================= METRICS =================

def compute_metrics(preds, refs):
    metrics = {
        "total": len(preds),
        "parse_errors": 0,
        "exact_match": 0,
        "field_metrics": defaultdict(lambda: {"correct": 0, "total": 0})
    }

    for p, r in zip(preds, refs):
        if p.get("parse_error"):
            metrics["parse_errors"] += 1
            continue

        if p == r:
            metrics["exact_match"] += 1

        for k in r:
            metrics["field_metrics"][k]["total"] += 1
            if k in p and p[k] == r[k]:
                metrics["field_metrics"][k]["correct"] += 1

    valid = metrics["total"] - metrics["parse_errors"]

    metrics["exact_match_rate"] = metrics["exact_match"] / max(valid, 1)
    metrics["parse_error_rate"] = metrics["parse_errors"] / metrics["total"]

    field_acc = {}
    for k, v in metrics["field_metrics"].items():
        field_acc[k] = v["correct"] / max(v["total"], 1)

    metrics["field_accuracy"] = field_acc
    metrics["avg_field_accuracy"] = (
        sum(field_acc.values()) / len(field_acc) if field_acc else 0
    )

    return metrics

# ================= MAIN =================

def evaluate(args):
    logger.info("🚀 Starting FIXED Evaluation")

    task, cl = setup_clearml()
    use_clearml = task is not None

    model, tokenizer = load_model(args.model_path)

    ds = load_dataset("json", data_files=args.test_data, split="train")

    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    schema_path = Path(args.test_data).parent / "extraction_schema.json"
    schema = json.load(open(schema_path))

    preds, refs = [], []

    for i in tqdm(range(0, len(ds), BATCH_SIZE)):
        batch = ds[i:i+BATCH_SIZE]

        notes = batch["input_text"]
        refs_batch = batch["silver_label"]

        preds_batch = batch_generate(model, tokenizer, notes, schema)

        preds.extend(preds_batch)
        refs.extend(refs_batch)

    metrics = compute_metrics(preds, refs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    with open(out / "predictions.jsonl", "w") as f:
        for p, r in zip(preds, refs):
            f.write(json.dumps({"prediction": p, "reference": r}) + "\n")

    if use_clearml:
        cl.report_single_value("exact_match", metrics["exact_match_rate"])
        cl.report_single_value("parse_error", metrics["parse_error_rate"])
        cl.report_single_value("avg_field_acc", metrics["avg_field_accuracy"])
        task.close()

    print("\n=== RESULTS ===")
    print(json.dumps(metrics, indent=2))


# ================= CLI =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int)

    args = parser.parse_args()
    evaluate(args)