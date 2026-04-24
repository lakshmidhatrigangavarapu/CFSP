#!/usr/bin/env python3
"""
DGX-Optimized Training Script: Clinical Factor Extraction
==========================================================
Optimized for 8x NVIDIA V100-32GB GPUs
Integrated with ClearML for full experiment tracking.

Project: Counterfactual Simulation of Extreme Mental Health Scenarios
Stage 1: Clinical Factor Extraction from MIMIC-IV Clinical Notes

This script fine-tunes BioMistral-7B using LoRA for extracting structured
clinical factors from mental health clinical notes.

ClearML Dashboard captures:
  - All hyper-parameters
  - Hardware / environment snapshot
  - Per-step: loss, grad_norm, learning-rate, GPU memory, CPU/RAM, throughput
  - Per-eval: eval_loss, perplexity
  - Per-epoch: average train loss
  - Best-model tracking (lowest eval_loss)
  - Model artifacts: best_model + latest_model
  - Dataset stats, extraction schema, training log
"""
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import load_dataset, Dataset
import torch
import psutil
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
import glob
import json
import sys
import os
os.environ["TRANSFORMERS_NO_SAFE_TENSORS_CONVERSION"] = "1"

# ============================================================================
# CLEARML INTEGRATION
# ============================================================================

from clearml import Task, Logger, OutputModel

# ClearML project / task names
CLEARML_PROJECT = "CFSP-DHATRI"
CLEARML_TASK_NAME = "BioMistral-7B-LoRA-ClinicalFactorExtraction"

# ============================================================================
# CONFIGURATION - Optimized for 8x V100-32GB
# ============================================================================

# Model
MODEL_ID = "BioMistral/BioMistral-7B"

# Paths
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
DATA_DIR = PACKAGE_DIR / "training_data"
OUTPUT_DIR = PACKAGE_DIR / "output"
LOG_FILE = OUTPUT_DIR / "training.log"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
LATEST_MODEL_DIR = OUTPUT_DIR / "latest_model"

# Sequence settings
MAX_SEQ_LEN = 2048
MAX_NOTE_CHARS = 6000

# Training settings - 8x V100
PER_DEVICE_BATCH = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
WARMUP_RATIO = 0.03

# LoRA settings
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Checkpointing
SAVE_STEPS = 500
LOGGING_STEPS = 10
EVAL_STEPS = 500

# Mixed precision
USE_FP16 = True
USE_BF16 = False

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging():
    """Configure logging for distributed training."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    handlers = []
    if is_main:
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode='a')
        ]

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CLEARML CALLBACK — logs everything possible to the dashboard
# ============================================================================

class ClearMLCallback(TrainerCallback):
    """
    Custom HuggingFace Trainer callback that pushes every available metric,
    GPU stat, system stat, and throughput number to ClearML scalars so
    they appear on the dashboard.
    """

    def __init__(self, task: Task):
        self.task = task
        self.clearml_logger: Logger = task.get_logger()
        self.best_eval_loss = float("inf")
        self.best_step = 0
        self.epoch_train_losses: list[float] = []
        self._training_start_time: Optional[datetime] = None

    # ---- helpers ----
    def _log_gpu_stats(self, step: int):
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            mem_peak = torch.cuda.max_memory_allocated(i) / 1e9
            self.clearml_logger.report_scalar(
                f"GPU-{i}/Memory", "Allocated (GB)", mem_alloc, step)
            self.clearml_logger.report_scalar(
                f"GPU-{i}/Memory", "Reserved (GB)", mem_reserved, step)
            self.clearml_logger.report_scalar(
                f"GPU-{i}/Memory", "Peak Allocated (GB)", mem_peak, step)

    def _log_system_stats(self, step: int):
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        self.clearml_logger.report_scalar(
            "System/CPU", "Utilisation %", cpu_pct, step)
        self.clearml_logger.report_scalar(
            "System/RAM", "Used (GB)", ram.used / 1e9, step)
        self.clearml_logger.report_scalar(
            "System/RAM", "Available (GB)", ram.available / 1e9, step)

    # ---- events ----
    def on_train_begin(self, args, state, control, **kw):
        self._training_start_time = datetime.now()
        logger.info("[ClearML] Training begin — logging started.")

    def on_log(self, args, state: TrainerState, control, logs=None, **kw):
        if logs is None:
            return
        step = state.global_step

        # --- standard HF metrics ---
        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("eval_"):
                title = "Eval/" + key.replace("eval_", "")
            elif key.startswith("train_"):
                title = "Train/" + key.replace("train_", "")
            else:
                title = "Train/" + key
            self.clearml_logger.report_scalar(title, key, value, step)

        # loss for epoch-level aggregation
        if "loss" in logs:
            self.epoch_train_losses.append(logs["loss"])

        # learning-rate scalar
        if "learning_rate" in logs:
            self.clearml_logger.report_scalar(
                "Schedule/LR", "learning_rate", logs["learning_rate"], step)

        # gradient norm
        if "grad_norm" in logs:
            self.clearml_logger.report_scalar(
                "Train/GradNorm", "grad_norm", logs["grad_norm"], step)

        # GPU + system
        self._log_gpu_stats(step)
        self._log_system_stats(step)

        # throughput
        if self._training_start_time:
            elapsed = (datetime.now() - self._training_start_time).total_seconds()
            if elapsed > 0:
                n_gpus = max(torch.cuda.device_count(), 1)
                samples = step * args.per_device_train_batch_size * n_gpus * args.gradient_accumulation_steps
                self.clearml_logger.report_scalar(
                    "Performance/Throughput", "samples_per_sec",
                    samples / elapsed, step)
                self.clearml_logger.report_scalar(
                    "Performance/Elapsed", "seconds", elapsed, step)

    def on_evaluate(self, args, state: TrainerState, control, metrics=None, **kw):
        if metrics is None:
            return
        step = state.global_step

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.clearml_logger.report_scalar(
                    "Eval/" + key.replace("eval_", ""), key, value, step)

        # perplexity (derived)
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            import math
            ppl = math.exp(min(eval_loss, 20))  # clamp to avoid overflow
            self.clearml_logger.report_scalar(
                "Eval/Perplexity", "perplexity", ppl, step)

        # best-model tracking
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = step
            logger.info(f"[ClearML] New best eval_loss={eval_loss:.6f} @ step {step}")
            self.clearml_logger.report_scalar(
                "Best/eval_loss", "best_eval_loss", eval_loss, step)

        self._log_gpu_stats(step)

    def on_epoch_end(self, args, state: TrainerState, control, **kw):
        epoch = int(state.epoch)
        step = state.global_step
        if self.epoch_train_losses:
            avg = sum(self.epoch_train_losses) / len(self.epoch_train_losses)
            self.clearml_logger.report_scalar(
                "Epoch/AvgTrainLoss", "avg_train_loss", avg, epoch)
            self.epoch_train_losses.clear()
        self.clearml_logger.report_scalar(
            "Epoch/Progress", "epoch", epoch, step)

    def on_save(self, args, state: TrainerState, control, **kw):
        self.clearml_logger.report_text(
            f"Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state: TrainerState, control, **kw):
        elapsed = 0
        if self._training_start_time:
            elapsed = (datetime.now() - self._training_start_time).total_seconds()
        self.clearml_logger.report_scalar(
            "Summary/TotalTime", "seconds", elapsed, state.global_step)
        self.clearml_logger.report_scalar(
            "Summary/BestEvalLoss", "best_eval_loss",
            self.best_eval_loss, state.global_step)
        self.clearml_logger.report_text(
            f"Training finished. Best eval_loss={self.best_eval_loss:.6f} "
            f"at step {self.best_step}")
        logger.info("[ClearML] Training end — all metrics flushed.")


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

SYSTEM_PROMPT = """You are a clinical AI assistant specialized in mental health. 
Extract structured clinical factors from the patient note according to the schema.
Output valid JSON only."""


def format_prompt(note: str, schema: Dict) -> str:
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS] + "... [truncated]"
    schema_str = json.dumps(schema, indent=2)
    return f"""<s>[INST] {SYSTEM_PROMPT}

### Clinical Note:
{note}

### Extraction Schema:
{schema_str}

### Instructions:
Extract all relevant clinical factors from the note. Return a JSON object matching the schema.
[/INST]"""


def format_example(example: Dict, schema: Dict) -> Dict[str, str]:
    prompt = format_prompt(example["note"], schema)
    labels_json = json.dumps(example["labels"], indent=2)
    full_text = f"{prompt}\n{labels_json}</s>"
    return {"text": full_text}

# ============================================================================
# DATA PREPARATION
# ============================================================================


def load_and_prepare_data(tokenizer) -> tuple:
    logger.info("Loading training data...")

    schema_path = DATA_DIR / "extraction_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"), split="train")

    logger.info(f"Train examples: {len(train_ds):,}")
    logger.info(f"Val examples: {len(val_ds):,}")

    def format_fn(examples):
        return {"text": [format_example({"note": n, "labels": l}, schema)["text"]
                         for n, l in zip(examples["input_text"], examples["silver_label"])]}

    train_ds = train_ds.map(format_fn, batched=True, batch_size=1000,
                            num_proc=1, remove_columns=train_ds.column_names,
                            desc="Formatting train data")
    val_ds = val_ds.map(format_fn, batched=True, batch_size=1000,
                        num_proc=1, remove_columns=val_ds.column_names,
                        desc="Formatting val data")

    return train_ds, val_ds, schema

# ============================================================================
# MODEL SETUP
# ============================================================================


def load_model_and_tokenizer():
    logger.info(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_safetensors=False,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer

# ============================================================================
# CHECKPOINT MANAGEMENT — keep only latest + best
# ============================================================================


def get_latest_checkpoint() -> Optional[str]:
    checkpoints = glob.glob(str(OUTPUT_DIR / "checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        logger.info(f"Found checkpoint: {latest}")
        return latest
    return None


def cleanup_checkpoints():
    """Remove all intermediate checkpoint-* folders so only
    best_model/ and latest_model/ remain."""
    for ckpt in glob.glob(str(OUTPUT_DIR / "checkpoint-*")):
        logger.info(f"Removing intermediate checkpoint: {ckpt}")
        shutil.rmtree(ckpt, ignore_errors=True)


def save_model_copy(trainer, tokenizer, dest: Path, tag: str, task: Task):
    """Save model + tokenizer and register as a ClearML OutputModel."""
    dest.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(dest))
    tokenizer.save_pretrained(str(dest))
    logger.info(f"Saved {tag} model to {dest}")

    output_model = OutputModel(task=task, name=tag, framework="PyTorch")
    output_model.update_weights(
        weights_filename=str(dest),
        auto_delete_file=False,
        target_filename=tag,
    )
    output_model.update_labels({"type": tag})
    logger.info(f"[ClearML] Registered {tag} as OutputModel")


# ============================================================================
# CLEARML INIT
# ============================================================================


def init_clearml_task() -> Task:
    """Create ClearML task and log every piece of context."""
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=CLEARML_TASK_NAME,
        auto_connect_frameworks={"tensorboard": True, "pytorch": True},
        reuse_last_task_id=False,
    )

    # --- hyper-parameters ---
    task.connect({
        "model_id": MODEL_ID,
        "max_seq_len": MAX_SEQ_LEN,
        "max_note_chars": MAX_NOTE_CHARS,
        "per_device_batch": PER_DEVICE_BATCH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "save_steps": SAVE_STEPS,
        "logging_steps": LOGGING_STEPS,
        "eval_steps": EVAL_STEPS,
        "fp16": USE_FP16,
        "bf16": USE_BF16,
    }, name="training_hyperparams")

    # --- hardware / environment ---
    hw = {
        "gpu_count": torch.cuda.device_count(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        hw[f"gpu_{i}_name"] = p.name
        hw[f"gpu_{i}_mem_gb"] = round(p.total_memory / 1e9, 1)
    task.connect(hw, name="hardware_info")

    # --- dataset stat files ---
    for sf in ("label_stats.json", "split_stats.json"):
        sp = DATA_DIR / sf
        if sp.exists():
            with open(sp) as f:
                task.connect(json.load(f), name=sf.replace(".json", ""))

    # --- upload extraction schema ---
    schema_path = DATA_DIR / "extraction_schema.json"
    if schema_path.exists():
        task.upload_artifact("extraction_schema",
                            artifact_object=str(schema_path))

    return task


# ============================================================================
# TRAINING
# ============================================================================


def train():
    """Main training function with full ClearML tracking and
    checkpoint-based resume support.  At the end only two models
    are kept: best_model (lowest eval_loss) and latest_model
    (last epoch)."""
    logger.info("=" * 60)
    logger.info("DGX Training: Clinical Factor Extraction")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"GPUs available: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f} GB)")

    # ---- ClearML (main process only) ----
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    task = init_clearml_task() if is_main else None

    # ---- Model ----
    model, tokenizer = load_model_and_tokenizer()

    if task:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        task.connect({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "lora_targets": "q,k,v,o,gate,up,down",
        }, name="model_info")

    # ---- Data ----
    train_ds, val_ds, schema = load_and_prepare_data(tokenizer)

    if task:
        task.connect({
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "schema_fields": list(schema.keys()) if isinstance(schema, dict) else "N/A",
        }, name="dataset_info")

    # ---- Schedule ----
    total_samples = len(train_ds)
    num_gpus = max(torch.cuda.device_count(), 1)
    eff_batch = PER_DEVICE_BATCH * num_gpus * GRAD_ACCUM_STEPS
    steps_per_epoch = total_samples // eff_batch
    total_steps = steps_per_epoch * NUM_EPOCHS

    logger.info("Training configuration:")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Per-device batch: {PER_DEVICE_BATCH}")
    logger.info(f"  Gradient accumulation: {GRAD_ACCUM_STEPS}")
    logger.info(f"  Effective batch size: {eff_batch}")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total steps: {total_steps:,}")

    if task:
        task.connect({
            "effective_batch_size": eff_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }, name="schedule_info")

    # ---- Trainer config ----
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        fp16=USE_FP16,
        bf16=USE_BF16,

        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Checkpoint strategy — Trainer keeps save_total_limit during run;
        # final cleanup afterwards leaves only best + latest.
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        logging_dir=str(OUTPUT_DIR / "runs"),
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to=["tensorboard"],

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,

        ddp_find_unused_parameters=False,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    # ---- Callbacks ----
    callbacks = []
    if task:
        callbacks.append(ClearMLCallback(task))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ---- Resume from checkpoint if available ----
    checkpoint = get_latest_checkpoint()

    logger.info("Starting training...")
    start_time = datetime.now()

    trainer.train(resume_from_checkpoint=checkpoint)

    train_time = datetime.now() - start_time
    logger.info(f"Training completed in {train_time}")

    # ==================================================================
    # SAVE EXACTLY TWO MODELS: best_model + latest_model
    # ==================================================================

    # 1) Best model (Trainer already loaded it via load_best_model_at_end)
    if task:
        save_model_copy(trainer, tokenizer, BEST_MODEL_DIR, "best_model", task)
    else:
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(BEST_MODEL_DIR))
        tokenizer.save_pretrained(str(BEST_MODEL_DIR))
    logger.info(f"Best model saved to {BEST_MODEL_DIR}")

    # 2) Latest model — from last checkpoint
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt and Path(latest_ckpt).exists():
        if LATEST_MODEL_DIR.exists():
            shutil.rmtree(LATEST_MODEL_DIR)
        shutil.copytree(latest_ckpt, str(LATEST_MODEL_DIR))
        tokenizer.save_pretrained(str(LATEST_MODEL_DIR))
        logger.info(f"Latest-epoch model saved to {LATEST_MODEL_DIR}")
        if task:
            om = OutputModel(task=task, name="latest_model", framework="PyTorch")
            om.update_weights(weights_filename=str(LATEST_MODEL_DIR),
                              auto_delete_file=False, target_filename="latest_model")
            om.update_labels({"type": "latest_model"})
    else:
        logger.info("No separate latest checkpoint; duplicating best as latest.")
        if LATEST_MODEL_DIR.exists():
            shutil.rmtree(LATEST_MODEL_DIR)
        shutil.copytree(str(BEST_MODEL_DIR), str(LATEST_MODEL_DIR))

    # 3) Cleanup intermediate checkpoints
    cleanup_checkpoints()

    # ==================================================================
    # FINAL SUMMARY → ClearML + local JSON
    # ==================================================================
    final_info = {
        "model_id": MODEL_ID,
        "training_time": str(train_time),
        "training_time_seconds": train_time.total_seconds(),
        "num_epochs": NUM_EPOCHS,
        "total_samples": total_samples,
        "effective_batch_size": eff_batch,
        "lora_rank": LORA_RANK,
        "max_seq_len": MAX_SEQ_LEN,
        "best_model_path": str(BEST_MODEL_DIR),
        "latest_model_path": str(LATEST_MODEL_DIR),
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "training_info.json", "w") as f:
        json.dump(final_info, f, indent=2)

    if task:
        task.upload_artifact("training_info",
                            artifact_object=str(OUTPUT_DIR / "training_info.json"))
        task.upload_artifact("training_log",
                            artifact_object=str(LOG_FILE))
        task.connect(final_info, name="final_summary")
        task.close()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Best  model -> {BEST_MODEL_DIR}")
    logger.info(f"  Latest model -> {LATEST_MODEL_DIR}")
    logger.info("=" * 60)

# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    train()
