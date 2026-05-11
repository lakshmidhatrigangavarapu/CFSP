# Counterfactual Mental Health Scenario Simulator

**Counterfactual Simulation of Extreme Mental Health Scenarios for Clinical Preparedness via Fine-Tuned LLMs and Explainable AI**

## Overview

A counterfactual scenario simulator for mental health analysis that generates possible extreme future scenarios using the current patient state via fine-tuned LLMs, with an XAI layer built on top – designed for clinical preparedness and psychology education.

### Key Features

- **Clinical Factor Extraction**: Extracts structured risk and protective factors from unstructured clinical notes using DSM-5/ICD-11 grounded schemas
- **Extreme Scenario Generation**: Generates counterfactual narratives of extreme adverse mental health trajectories
- **Causal Explainability**: XAI layer providing causal pathway justifications and uncertainty estimates
- **Second Reader Paradigm**: Acts as a consulting second opinion, not a final decision maker

## Project Structure

```
├── full_pipeline/          # Pipeline v1 - Core implementation
├── full_pipeline_v2/       # Pipeline v2 - Enhanced features
├── full_pipeline_v3/       # Pipeline v3 - Optimizations
├── full_pipeline_v4/       # Pipeline v4 - Latest version
├── input-output/           # Sample input/output examples
├── output_v2/              # Model outputs and results
├── Paper/                  # Research paper and documentation
├── scripts/                # Training and utility scripts
├── training_data/          # Training datasets
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA-compatible GPU (8x NVIDIA V100-32GB recommended for training)
- 256GB+ system RAM for training

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/counterfactual-mental-health.git
cd counterfactual-mental-health

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Setup Environment
```bash
./scripts/setup_env.sh
```

### 2. Run Training
```bash
./scripts/run_training.sh
```

### 3. Run Inference Pipeline
```bash
cd full_pipeline_v4
python run.py --input your_clinical_notes.jsonl
```

## Pipeline Versions

| Version | Description |
|---------|-------------|
| v1 (`full_pipeline/`) | Initial implementation with core factor extraction |
| v2 (`full_pipeline_v2/`) | Added scenario generation capabilities |
| v3 (`full_pipeline_v3/`) | Performance optimizations |
| v4 (`full_pipeline_v4/`) | Latest with XAI integration |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | BioMistral/BioMistral-7B |
| Precision | FP16 |
| Per-GPU Batch | 8 |
| Gradient Accumulation | 2 |
| Sequence Length | 2048 |
| LoRA Rank | 64 |
| Epochs | 3 |
| Learning Rate | 2e-4 |

## Components

### Clinical Factor Extraction
Extracts structured clinical factors from unstructured mental health notes:
- Risk factors
- Protective factors
- Current symptoms
- Historical patterns

### Extreme Scenario Generation
Generates counterfactual narratives targeting boundary conditions:
- Worst-case trajectory projection
- Multiple scenario pathways
- Clinically grounded outputs

### Causal Explainability (XAI)
Provides interpretable explanations:
- Causal pathway visualization
- Factor contribution analysis
- Confidence/uncertainty estimates

## Evaluation

```bash
python scripts/evaluate.py \
    --model_path output/final_model \
    --test_data training_data/test.jsonl \
    --output_dir output/eval_results
```

## Hardware Requirements

### For Training
- 8x NVIDIA V100-32GB GPUs (or equivalent)
- NVLink interconnect (recommended)
- 256GB+ system RAM
- Fast NVMe storage

### For Inference
- 1x GPU with 16GB+ VRAM
- 32GB system RAM

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is designed for clinical preparedness and educational purposes only. It is not intended as a diagnostic tool or replacement for professional clinical judgment. Always consult qualified mental health professionals for patient care decisions.

## Acknowledgments

- Built using HuggingFace Transformers
- Fine-tuned on BioMistral-7B
- Validated against clinician-annotated gold standards
