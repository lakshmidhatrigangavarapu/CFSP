# Counterfactual Mental Health Scenario Simulator

**Counterfactual Simulation of Extreme Mental Health Scenarios for Clinical Preparedness via Fine-Tuned LLMs and Explainable AI**

**Counterfactual Extreme Mental Health Scenario Simulator Interface**

Application Interface:
<img width="3360" height="2100" alt="image" src="https://github.com/user-attachments/assets/728627a0-6e7e-4766-9491-faefb09f822d" />

Results generated:
<img width="3360" height="2100" alt="image" src="https://github.com/user-attachments/assets/5f15c310-d93b-42e1-8d31-72550768ce3b" />
Path of the generated pdf : CFSP/input-output/report_v4_ui_ui.pdf


A safety-aware fine-tuned LLM framework that acts as a **second reader** of mental health documentation — not to predict outcomes, but to **prepare clinicians and psychology students** for extreme patient trajectories.

## Overview

The framework operates in three stages:

1. **Phase 1: Clinical Factor Extraction** — Extracts structured risk and protective factors from unstructured clinical notes using a DSM-5 and ICD-11 grounded schema
2. **Phase 2: Extreme Scenario Generation** — Generates counterfactual narratives of extreme adverse mental health trajectories using Tree-of-Thoughts reasoning
3. **Phase 3: Clinical Report with XAI** — Generates patient preparedness reports with causal pathway justifications and uncertainty estimates

## Quick Start

### Training

```bash
# 1. Setup environment
./scripts/setup_env.sh

# 2. Activate environment
source dgx_env/bin/activate

# 3. Launch training
./scripts/run_training.sh
```

### Inference

```bash
# Run the full pipeline on a clinical note
source dgx_env/bin/activate
python -m full_pipeline_v4 --input "path/to/clinical_note.txt"

# Or use the Gradio web interface
python full_pipeline_v4/app.py
```

## Package Structure

```
dgx_package/
├── README.md                    # This file
├── training_data/               # Training data
│   ├── train.jsonl             # 94,592 training examples
│   ├── val.jsonl               # Validation set
│   ├── test.jsonl              # Test set
│   └── extraction_schema.json  # DSM-5/ICD-11 grounded schema
├── scripts/
│   ├── setup_env.sh            # Environment setup
│   ├── train_dgx.py            # Main training script
│   └── run_training.sh         # Training launcher
├── full_pipeline_v4/           # Inference pipeline
│   ├── app.py                  # Gradio web interface
│   ├── run.py                  # CLI runner
│   ├── pipeline.py             # Main pipeline orchestrator
│   ├── silver_label_extractor.py  # Phase 1: Factor extraction
│   ├── scenario_generator.py   # Phase 2: ToT scenario generation
│   ├── report_generator.py     # Phase 3: Clinical report
│   ├── evidence_attribution.py # XAI layer
│   └── config.py               # Pipeline configuration
├── output_v2/                  # Trained model checkpoints
│   ├── best_model/             # Best checkpoint (by eval loss)
│   └── latest_model/           # Final checkpoint
└── clearml_experiment_data/    # Training experiment logs
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | microsoft/Phi-3.5-mini-instruct | 3.8B parameters |
| Fine-tuning | LoRA (rank 32) | Parameter-efficient |
| Sequence Length | 2048 | Full context window |
| Epochs | 5 | Full training |
| Effective Batch Size | 64 | Multi-GPU distributed |
| Training Time | ~35 hours | 4x V100 GPUs |
| Training Samples | 94,592 | MIMIC-IV derived |

## Inference Pipeline

### Generation Parameters

| Phase | Max Tokens | Temperature | Purpose |
|-------|------------|-------------|---------|
| Phase 1 | 1536 | 0.0 | Deterministic extraction |
| Phase 2 Step 1 | 512 | 0.3 | Trigger identification |
| Phase 2 Step 2 | 512 | 0.3 | Causal chain reasoning |
| Phase 2 Step 3 | 1280 | 0.3 | Scenario narrative |
| Phase 3 | 2048 | 0.2 | Clinical report |

### Scenario Branches

The system generates three parallel scenario branches:
- **Decompensation** — Psychiatric deterioration trajectory
- **Crisis** — Acute crisis escalation trajectory  
- **Recovery Failure** — Treatment non-response trajectory

Branch gating skips irrelevant branches based on extracted clinical factors.

## Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# View training logs
tail -f output_v2/training.log

# TensorBoard
tensorboard --logdir output_v2/runs
```

## Troubleshooting

### OOM Errors
- Reduce batch size in training config
- Gradient checkpointing is enabled by default

### NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

## Citation

If using this pipeline:
```
Johnson, A., et al. (2023). MIMIC-IV (version 3.1). PhysioNet.
```
