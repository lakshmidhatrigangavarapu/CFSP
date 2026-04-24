"""
Interactive Gradio UI for the Full Inference Pipeline
======================================================
Launch with:
    python -m full_pipeline.app
    python -m full_pipeline.app --gpu 4
    python -m full_pipeline.app --gpu 0,1 --port 7860

The model is loaded ONCE when you click "Load Model".
Re-click "Load Model" only when you change GPU / checkpoint settings.
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from full_pipeline.config import BASE_MODEL_ID, CHECKPOINT_PATH, NUM_BRANCHES, OUTPUT_DIR
from full_pipeline.model_loader import load_model
from full_pipeline.pipeline import run_pipeline
from full_pipeline.report_generator import format_report_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_pipeline.app")

# ---------------------------------------------------------------------------
# Known JSONL datasets (relative to package root)
# Files larger than this will use streaming random selection
# ---------------------------------------------------------------------------
DATASETS = {
    "one_sample.jsonl (1 record)": str(PACKAGE_DIR / "one_sample.jsonl"),
    "one_sample_test10.jsonl (10 records)": str(PACKAGE_DIR / "one_sample_test10.jsonl"),
    "training_data/val.jsonl": str(PACKAGE_DIR / "training_data" / "val.jsonl"),
    "training_data/test.jsonl": str(PACKAGE_DIR / "training_data" / "test.jsonl"),
}
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB — use streaming above this

# ---------------------------------------------------------------------------
# Global model state (loaded once, reused across runs)
# ---------------------------------------------------------------------------
_model_state = {"model": None, "tokenizer": None, "device": None, "loaded_with": None}


def _model_loaded() -> bool:
    return _model_state["model"] is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_line_from_large_file(filepath: str) -> str:
    """Reservoir sampling (k=1) for huge files — O(n) time, O(1) memory."""
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
    """Return a random JSONL record from filepath (handles large files)."""
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


def _count_lines(filepath: str) -> int:
    """Count non-empty lines in a file efficiently."""
    path = Path(filepath)
    if not path.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _format_silver_labels(labels: dict) -> str:
    return json.dumps(labels, indent=2)


def _format_scenarios(scenarios_result: dict) -> str:
    return json.dumps(scenarios_result, indent=2)


def _format_full_result(result: dict) -> str:
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def load_model_action(gpu_ids: str, base_model: str, checkpoint: str, base_only: bool):
    """Load / reload the model. Returns a status message."""
    global _model_state

    key = (gpu_ids, base_model, checkpoint, base_only)
    if _model_loaded() and _model_state["loaded_with"] == key:
        return "✅ Model already loaded with these settings."

    status_msg = f"Loading model on GPU(s): {gpu_ids} ..."
    log.info(status_msg)

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
        log.error(f"Model loading failed: {e}")
        return f"❌ Load failed:\n{traceback.format_exc()}"


def pick_random_record(dataset_name: str):
    """Pick a random record from the selected dataset. Returns (note_text, json_text, info_md)."""
    filepath = DATASETS.get(dataset_name)
    if not filepath:
        return "", "", "Unknown dataset."

    record = _random_record(filepath)
    if record is None:
        return "", "", f"⚠️ Could not read a record from: {filepath}"

    note_text = record.get("input_text", "")
    subject_id = record.get("subject_id", "?")
    hadm_id = record.get("hadm_id", "?")

    # Build info
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
    """
    Run inference. Returns (report_text, silver_labels_json, scenarios_json, full_json, status).
    input_mode: "Text" | "JSON" | "Random"
    """
    if not _model_loaded():
        err = "❌ Model not loaded. Click 'Load Model' first."
        return err, "", "", "", err

    # --- Resolve note text and optional pre-existing labels ---
    note_text = ""
    existing_labels = None
    subject_id = "ui"
    hadm_id = "ui"

    if input_mode == "Text":
        note_text = text_input.strip()
        if not note_text:
            return "❌ Please paste a clinical note in the Text tab.", "", "", "", "No input."

    elif input_mode == "JSON":
        raw = json_input.strip()
        if not raw:
            return "❌ Please paste a JSON record in the JSON tab.", "", "", "", "No input."
        try:
            # Accept both a plain JSON object and a JSONL line
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
            return "❌ No random record loaded. Click 'Pick Random Record' first.", "", "", "", "No input."

    if not note_text:
        return "❌ Could not extract note text from input.", "", "", "", "Empty note."

    try:
        log.info(f"Running pipeline | subject={subject_id} hadm={hadm_id} | branches={num_branches}")

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

        # Format human-readable report
        report_txt = format_report_text(
            report=result["phase3_report"],
            silver_labels=result["phase1_silver_labels"],
            scenarios_result=result["phase2_scenarios"],
        )

        silver_json = _format_silver_labels(result["phase1_silver_labels"])
        scenarios_json = _format_scenarios(result["phase2_scenarios"])
        full_json = _format_full_result(result)

        duration = result.get("duration_seconds", "?")
        status = f"✅ Done in {duration}s — subject={subject_id} hadm={hadm_id}"

        return report_txt, silver_json, scenarios_json, full_json, status

    except Exception as e:
        err_detail = traceback.format_exc()
        log.error(f"Pipeline error: {err_detail}")
        return f"❌ Pipeline failed:\n{err_detail}", "", "", "", f"Error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui(default_gpu: str = "0"):
    with gr.Blocks(title="Clinical Pipeline Tester", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🏥 Clinical Pipeline Tester")
        gr.Markdown(
            "Load the model once, then run the full 3-phase pipeline on any clinical note. "
            "Input can be raw text, a JSON record, or a random sample from your datasets."
        )

        # ===================================================================
        # ROW 1 — Model Settings
        # ===================================================================
        with gr.Row():
            with gr.Group():
                gr.Markdown("### ⚙️ Model Settings")
                with gr.Row():
                    gpu_box = gr.Textbox(
                        value=default_gpu,
                        label="GPU(s)",
                        placeholder="e.g. 0  or  1,2,3",
                        scale=1,
                    )
                    base_model_box = gr.Textbox(
                        value=BASE_MODEL_ID,
                        label="Base Model",
                        scale=3,
                    )
                with gr.Row():
                    checkpoint_box = gr.Textbox(
                        value=CHECKPOINT_PATH,
                        label="Checkpoint Path (LoRA)",
                        scale=3,
                    )
                    base_only_chk = gr.Checkbox(
                        value=False,
                        label="Base model only (no LoRA)",
                        scale=1,
                    )
                with gr.Row():
                    load_btn = gr.Button("🔄 Load Model", variant="primary", scale=1)
                    model_status = gr.Textbox(
                        value="(not loaded)",
                        label="Model Status",
                        interactive=False,
                        scale=3,
                    )

        load_btn.click(
            fn=load_model_action,
            inputs=[gpu_box, base_model_box, checkpoint_box, base_only_chk],
            outputs=[model_status],
        )

        gr.Markdown("---")

        # ===================================================================
        # ROW 2 — Input + Controls + Output
        # ===================================================================
        with gr.Row(equal_height=False):

            # ---------------------------------------------------------------
            # LEFT — Input panel
            # ---------------------------------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Input")

                input_mode = gr.Radio(
                    choices=["Text", "JSON", "Random"],
                    value="Text",
                    label="Input mode",
                )

                # Text tab
                with gr.Group(visible=True) as text_group:
                    text_input = gr.Textbox(
                        label="Paste clinical note (plain text)",
                        lines=18,
                        placeholder="Paste the discharge summary or clinical note here...",
                    )

                # JSON tab
                with gr.Group(visible=False) as json_group:
                    json_input = gr.Textbox(
                        label="Paste JSONL record (JSON object with 'input_text' key)",
                        lines=18,
                        placeholder='{"subject_id": "123", "input_text": "...", "silver_label": {...}}',
                    )

                # Random tab
                with gr.Group(visible=False) as random_group:
                    dataset_dd = gr.Dropdown(
                        choices=list(DATASETS.keys()),
                        value=list(DATASETS.keys())[0],
                        label="Dataset",
                    )
                    pick_btn = gr.Button("🎲 Pick Random Record")
                    random_info = gr.Markdown("_(no record loaded)_")
                    random_note = gr.Textbox(
                        label="Selected note (read-only preview)",
                        lines=12,
                        interactive=False,
                    )
                    random_json_hidden = gr.State("")  # full JSON for JSON mode

                pick_btn.click(
                    fn=pick_random_record,
                    inputs=[dataset_dd],
                    outputs=[random_note, random_json_hidden, random_info],
                )

                gr.Markdown("---")
                gr.Markdown("### ▶️ Run Settings")
                with gr.Row():
                    branches_slider = gr.Slider(
                        minimum=2, maximum=3, step=1, value=NUM_BRANCHES,
                        label="ToT Branches",
                        scale=1,
                    )
                    use_labels_chk = gr.Checkbox(
                        value=False,
                        label="Re-use silver labels from JSON (skip Phase 1)",
                        scale=2,
                    )
                run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
                run_status = gr.Textbox(
                    label="Run Status",
                    interactive=False,
                    lines=2,
                )

            # ---------------------------------------------------------------
            # RIGHT — Output panel
            # ---------------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Output")
                with gr.Tabs():
                    with gr.Tab("📄 Report"):
                        report_out = gr.Textbox(
                            label="Clinical Preparedness Report",
                            lines=40,
                            interactive=False,
                        )
                    with gr.Tab("🏷️ Silver Labels (Phase 1)"):
                        labels_out = gr.Code(
                            label="Extracted Silver Labels (JSON)",
                            language="json",
                            lines=40,
                            interactive=False,
                        )
                    with gr.Tab("🌲 Scenarios (Phase 2)"):
                        scenarios_out = gr.Code(
                            label="ToT Scenarios (JSON)",
                            language="json",
                            lines=40,
                            interactive=False,
                        )
                    with gr.Tab("📦 Full Result"):
                        full_out = gr.Code(
                            label="Complete Pipeline Result (JSON)",
                            language="json",
                            lines=40,
                            interactive=False,
                        )

        # -------------------------------------------------------------------
        # Visibility toggle for input groups
        # -------------------------------------------------------------------
        def toggle_input_groups(mode):
            return (
                gr.update(visible=(mode == "Text")),
                gr.update(visible=(mode == "JSON")),
                gr.update(visible=(mode == "Random")),
            )

        input_mode.change(
            fn=toggle_input_groups,
            inputs=[input_mode],
            outputs=[text_group, json_group, random_group],
        )

        # When a random record is picked in JSON mode, pre-populate JSON box too
        def sync_random_to_json(json_str):
            return json_str

        random_json_hidden.change(
            fn=sync_random_to_json,
            inputs=[random_json_hidden],
            outputs=[json_input],
        )

        # -------------------------------------------------------------------
        # Run button — resolve which note to use based on current mode
        # -------------------------------------------------------------------
        run_btn.click(
            fn=run_pipeline_action,
            inputs=[
                input_mode,
                text_input,
                json_input,
                random_note,
                use_labels_chk,
                branches_slider,
            ],
            outputs=[report_out, labels_out, scenarios_out, full_out, run_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the clinical pipeline UI")
    parser.add_argument("--gpu", default="0", help="GPU index(es), e.g. '4' or '0,1'")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_ui(default_gpu=args.gpu)
    log.info(f"Starting UI on {args.host}:{args.port} ...")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
