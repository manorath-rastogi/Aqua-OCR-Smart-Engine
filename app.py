"""
Aqua OCR Smart Engine (AQIVON OmniOCR) — Clean & Minimal Testing Interface.
Unified OCR interface supporting:
- Document / Image Upload (PNG, JPG, PDF)
- Language Selection (Auto, English, Hindi, Marathi, Urdu, Bengali, etc.)
- OCR Engine Selection (Auto, Tesseract, PaddleOCR, IndicOCR)
- Direct real-backend execution with copy-ready text extraction
"""

import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paddle environment flags to ensure flawless Windows CPU/GPU runtime
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import gradio as gr
from aqua_ocr_smart_engine import IntelligentRouter
from aqua_ocr_smart_engine.types import OCRResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqua_ocr_smart_engine.app")

# Initialize global intelligent router connected to real OCR engines
ROUTER = IntelligentRouter()
SAMPLES_DIR = PROJECT_ROOT / "samples"

LANGUAGE_MAP = {
    "Auto": "auto",
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Urdu": "ur",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Odia": "or",
}

ENGINE_MAP = {
    "Auto": None,
    "Tesseract": "tesseract",
    "PaddleOCR": "paddleocr",
    "IndicOCR": "indicocr",
}


def run_ocr(
    file_path: Optional[str],
    language_display: str,
    engine_display: str
) -> Tuple[str, str]:
    """
    Executes actual OCR pipeline via Aqua OCR Smart Engine.
    Returns:
        Tuple of (extracted_text, status_message)
    """
    if not file_path or not os.path.exists(file_path):
        return (
            "",
            "⚠️ Error: Please upload a valid document or image before clicking 'Run OCR'."
        )

    lang_code = LANGUAGE_MAP.get(language_display, "auto")
    engine_id = ENGINE_MAP.get(engine_display, None)
    file_name = Path(file_path).name

    start_time = time.time()
    try:
        logger.info(
            "Processing '%s' [Engine: %s, Lang: %s]",
            file_name, engine_display, language_display
        )

        ocr_result: OCRResult = ROUTER.route_and_process(
            document_source=file_path,
            requested_engine=engine_id,
            language=lang_code,
            document_name=file_name
        )

        elapsed_sec = time.time() - start_time

        # Extract text: if single page, keep clean format; if multi-page, keep page dividers
        if len(ocr_result.pages) == 1:
            extracted_text = ocr_result.pages[0].text.strip()
        else:
            extracted_text = ocr_result.full_text.strip()

        if not extracted_text:
            extracted_text = "(No text detected in this document)"

        # Prepare clear diagnostic status message
        engine_used_name = ocr_result.metadata.get("engine_name") or ocr_result.engine_used
        total_blocks = sum(len(p.blocks) for p in ocr_result.pages)
        device = ocr_result.metadata.get("device", "CPU")

        status_msg = (
            f"✅ OCR Completed in {elapsed_sec:.2f}s | "
            f"Engine Used: {engine_used_name} | "
            f"Pages: {ocr_result.num_pages} | "
            f"Detected Blocks: {total_blocks} | "
            f"Device: {device}"
        )

        # Notify if Tesseract fell back to PaddleOCR
        if engine_display == "Tesseract" and ocr_result.engine_used != "tesseract":
            status_msg += " (Note: Tesseract binary not found on system PATH; automatically processed via PaddleOCR)"

        return extracted_text, status_msg

    except Exception as exc:
        logger.exception("OCR execution error on '%s': %s", file_name, exc)
        return (
            "",
            f"❌ OCR Error: {type(exc).__name__} - {str(exc)}"
        )


def build_interface() -> gr.Blocks:
    """Creates a minimal, clean, and functional Gradio UI."""
    with gr.Blocks(title="Aqua OCR Smart Engine") as demo:
        gr.Markdown(
            """
            # Aqua OCR Smart Engine
            **Unified Modular OCR Engine** &bull; AQIVON Technologies
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload Document / Image (PNG, JPG, PDF)",
                    file_types=["image", ".pdf"],
                    type="filepath"
                )

                language_selector = gr.Dropdown(
                    label="Language Selection",
                    choices=list(LANGUAGE_MAP.keys()),
                    value="Auto",
                    interactive=True
                )

                engine_selector = gr.Dropdown(
                    label="OCR Engine Selection",
                    choices=list(ENGINE_MAP.keys()),
                    value="Auto",
                    interactive=True
                )

                run_btn = gr.Button("Run OCR", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Extracted Text (with copy option)",
                    placeholder="Extracted text will appear here...",
                    lines=16,
                    buttons=["copy"],
                    interactive=False
                )

                status_box = gr.Textbox(
                    label="Status & Execution Diagnostics",
                    value="Ready. Please upload a document and click 'Run OCR'.",
                    lines=2,
                    interactive=False
                )

        # Trigger OCR on click
        run_btn.click(
            fn=run_ocr,
            inputs=[input_file, language_selector, engine_selector],
            outputs=[output_text, status_box]
        )

        # Preloaded samples for one-click verification
        example_samples = []
        sample_invoice = SAMPLES_DIR / "sample_english_invoice.png"
        sample_hindi = SAMPLES_DIR / "sample_indic_hindi.png"

        if sample_invoice.exists():
            example_samples.append([str(sample_invoice), "English", "PaddleOCR"])
        if sample_hindi.exists():
            example_samples.append([str(sample_hindi), "Hindi", "IndicOCR"])

        if example_samples:
            gr.Examples(
                examples=example_samples,
                inputs=[input_file, language_selector, engine_selector],
                outputs=[output_text, status_box],
                fn=run_ocr,
                cache_examples=False,
                label="Quick Test Samples"
            )

    return demo


if __name__ == "__main__":
    app = build_interface()
    # Launch on localhost
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
