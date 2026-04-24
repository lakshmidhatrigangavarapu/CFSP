"""
Interactive Gradio UI for the Full Inference Pipeline V2
=========================================================
Launch with:
    python -m full_pipeline_v2.app
    python -m full_pipeline_v2.app --gpu 4
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline_v2.config import BASE_MODEL_ID, CHECKPOINT_PATH, NUM_BRANCHES, OUTPUT_DIR
from full_pipeline_v2.model_loader import load_model
from full_pipeline_v2.pipeline import run_pipeline
from full_pipeline_v2.report_generator import format_report_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_pipeline_v2.app")

DATASETS = {
    "one_sample.jsonl (1 record)": str(PACKAGE_DIR / "one_sample.jsonl"),
    "one_sample_test10.jsonl (10 records)": str(PACKAGE_DIR / "one_sample_test10.jsonl"),
    "training_data/val.jsonl": str(PACKAGE_DIR / "training_data" / "val.jsonl"),
    "training_data/test.jsonl": str(PACKAGE_DIR / "training_data" / "test.jsonl"),
}
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024

_model_state = {"model": None, "tokenizer": None, "device": None, "loaded_with": None}


def _model_loaded() -> bool:
    return _model_state["model"] is not None


def _random_line_from_large_file(filepath: str) -> str:
    chosen = None
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            if random.randint(1, count) == 1:
                chosen = line
    return chosen


def _random_record(filepath: str) -> Optional[dict]:
    path = Path(filepath)
    if not path.exists():
        return None
    if path.stat().st_size > LARGE_FILE_THRESHOLD:
        line = _random_line_from_large_file(filepath)
    else:
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return None
        line = random.choice(lines)
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def load_model_action(gpu_ids: str, base_model: str, checkpoint: str, base_only: bool):
    global _model_state
    key = (gpu_ids, base_model, checkpoint, base_only)
    if _model_loaded() and _model_state["loaded_with"] == key:
        return "✅ Model already loaded with these settings."
    try:
        multi_gpu = "," in gpu_ids
        load_adapters = not base_only
        if multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            gpu_id = 0
        else:
            gpu_id = int(gpu_ids.strip())
        model, tokenizer, device = load_model(
            gpu_id=gpu_id,
            base_model_id=base_model,
            checkpoint_path=checkpoint if checkpoint.strip() else None,
            load_adapters=load_adapters,
            multi_gpu=multi_gpu,
        )
        _model_state["model"] = model
        _model_state["tokenizer"] = tokenizer
        _model_state["device"] = device
        _model_state["loaded_with"] = key
        adapter_info = f"LoRA from {checkpoint}" if load_adapters else "Base model only"
        return f"✅ Model loaded on {device}.\n   {base_model}\n   {adapter_info}"
    except Exception as e:
        _model_state["model"] = None
        return f"❌ Load failed:\n{traceback.format_exc()}"


def pick_random_record(dataset_name: str):
    filepath = DATASETS.get(dataset_name)
    if not filepath:
        return "", "", "Unknown dataset."
    record = _random_record(filepath)
    if record is None:
        return "", "", f"⚠️ Could not read a record from: {filepath}"
    note_text = record.get("input_text", "")
    subject_id = record.get("subject_id", "?")
    hadm_id = record.get("hadm_id", "?")
    has_labels = bool(record.get("silver_label"))
    info = (
        f"**Subject:** {subject_id} | **Admission:** {hadm_id}  \n"
        f"**Note length:** {len(note_text):,} chars  \n"
        f"**Has pre-extracted silver labels:** {'Yes' if has_labels else 'No'}"
    )
    return note_text, json.dumps(record, indent=2), info


def run_pipeline_action(
    input_mode: str,
    text_input: str,
    json_input: str,
    random_note: str,
    use_existing_labels: bool,
    num_branches: int,
):
    if not _model_loaded():
        err = "❌ Model not loaded. Click 'Load Model' first."
        return err, "", "", "", err

    note_text = ""
    existing_labels = None
    subject_id = "ui"
    hadm_id = "ui"

    if input_mode == "Text":
        note_text = text_input.strip()
        if not note_text:
            return "❌ Please paste a clinical note.", "", "", "", "No input."
    elif input_mode == "JSON":
        raw = json_input.strip()
        if not raw:
            return "❌ Please paste a JSON record.", "", "", "", "No input."
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"❌ JSON parse error: {e}", "", "", "", "Parse error."
        note_text = record.get("input_text", "")
        subject_id = str(record.get("subject_id", "ui"))
        hadm_id = str(record.get("hadm_id", "ui"))
        if use_existing_labels:
            existing_labels = record.get("silver_label") or record.get("extracted_factors")
    elif input_mode == "Random":
        note_text = random_note.strip()
        if not note_text:
            return "❌ No random record loaded.", "", "", "", "No input."

    if not note_text:
        return "❌ Could not extract note text.", "", "", "", "Empty note."

    try:
        result = run_pipeline(
            report_text=note_text,
            model=_model_state["model"],
            tokenizer=_model_state["tokenizer"],
            device=_model_state["device"],
            num_branches=int(num_branches),
            silver_labels=existing_labels,
            use_finetuned=True,
        )
        result["subject_id"] = subject_id
        result["hadm_id"] = hadm_id

        critical_signals = result.get("phase1_6_critical_signals", {})
        report_txt = format_report_text(
            report=result["phase3_report"],
            silver_labels=result["phase1_silver_labels"],
            scenarios_result=result["phase2_scenarios"],
            critical_signals=critical_signals,
        )

        silver_json = json.dumps(result["phase1_silver_labels"], indent=2)
        scenarios_json = json.dumps(result["phase2_scenarios"], indent=2)
        full_json = json.dumps(result, indent=2)

        duration = result.get("duration_seconds", "?")
        # Build status with V2 info
        norm_count = len(result.get("phase1_5_normalization_log", []))
        signal_found = "YES" if critical_signals.get("signals_found") else "no"
        gated = sum(1 for b in result.get("phase1_7_branch_gating", []) if b.get("gated"))
        status = (
            f"✅ Done in {duration}s | "
            f"Normalizer: {norm_count} corrections | "
            f"Critical signals: {signal_found} | "
            f"Gated branches: {gated}"
        )

        return report_txt, silver_json, scenarios_json, full_json, status

    except Exception as e:
        err_detail = traceback.format_exc()
        return f"❌ Pipeline failed:\n{err_detail}", "", "", "", f"Error: {e}"


def build_ui(default_gpu: str = "0"):
    with gr.Blocks(title="Clinical Pipeline V2 Tester", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🏥 Clinical Pipeline V2 Tester")
        gr.Markdown(
            "V2 improvements: post-extraction normalization, critical signal scanning, "
            "conditional branch gating, evidence binding, final consistency check."
        )

        with gr.Row():
            with gr.Group():
                gr.Markdown("### ⚙️ Model Settings")
                with gr.Row():
                    gpu_box = gr.Textbox(value=default_gpu, label="GPU(s)", scale=1)
                    base_model_box = gr.Textbox(value=BASE_MODEL_ID, label="Base Model", scale=3)
                with gr.Row():
                    checkpoint_box = gr.Textbox(value=CHECKPOINT_PATH, label="Checkpoint (LoRA)", scale=3)
                    base_only_chk = gr.Checkbox(value=False, label="Base model only", scale=1)
                with gr.Row():
                    load_btn = gr.Button("🔄 Load Model", variant="primary", scale=1)
                    model_status = gr.Textbox(value="(not loaded)", label="Status", interactive=False, scale=3)

        load_btn.click(fn=load_model_action, inputs=[gpu_box, base_model_box, checkpoint_box, base_only_chk], outputs=[model_status])

        gr.Markdown("---")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Input")
                input_mode = gr.Radio(choices=["Text", "JSON", "Random"], value="Text", label="Input mode")

                with gr.Group(visible=True) as text_group:
                    text_input = gr.Textbox(label="Clinical note (plain text)", lines=18)

                with gr.Group(visible=False) as json_group:
                    json_input = gr.Textbox(label="JSONL record", lines=18)

                with gr.Group(visible=False) as random_group:
                    dataset_dd = gr.Dropdown(choices=list(DATASETS.keys()), value=list(DATASETS.keys())[0], label="Dataset")
                    pick_btn = gr.Button("🎲 Pick Random Record")
                    random_info = gr.Markdown("_(no record loaded)_")
                    random_note = gr.Textbox(label="Selected note", lines=12, interactive=False)

                pick_btn.click(fn=pick_random_record, inputs=[dataset_dd], outputs=[random_note, gr.State(""), random_info])

                def toggle_input(mode):
                    return (
                        gr.update(visible=mode == "Text"),
                        gr.update(visible=mode == "JSON"),
                        gr.update(visible=mode == "Random"),
                    )

                input_mode.change(fn=toggle_input, inputs=[input_mode], outputs=[text_group, json_group, random_group])

                gr.Markdown("---")
                gr.Markdown("### ▶️ Run Settings")
                with gr.Row():
                    branches_slider = gr.Slider(minimum=2, maximum=3, step=1, value=NUM_BRANCHES, label="ToT Branches", scale=1)
                    use_labels_chk = gr.Checkbox(value=False, label="Re-use silver labels", scale=2)
                run_btn = gr.Button("🚀 Run Pipeline V2", variant="primary", size="lg")
                run_status = gr.Textbox(label="Run Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Output")
                with gr.Tabs():
                    with gr.Tab("📄 Report"):
                        report_out = gr.Textbox(label="Clinical Preparedness Report (V2)", lines=40, interactive=False)
                    with gr.Tab("🏷️ Silver Labels"):
                        labels_out = gr.Code(label="Extracted + Normalized Silver Labels", language="json", lines=40, interactive=False)
                    with gr.Tab("🌲 Scenarios"):
                        scenarios_out = gr.Code(label="ToT Scenarios (with gating)", language="json", lines=40, interactive=False)
                    with gr.Tab("📦 Full JSON"):
                        full_out = gr.Code(label="Complete Pipeline Output", language="json", lines=40, interactive=False)

        run_btn.click(
            fn=run_pipeline_action,
            inputs=[input_mode, text_input, json_input, random_note, use_labels_chk, branches_slider],
            outputs=[report_out, labels_out, scenarios_out, full_out, run_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Clinical Pipeline V2 Gradio UI")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(default_gpu=args.gpu)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
