"""
AQIVON OmniOCR - Lightweight Testing Interface.
Strictly a verification and diagnostic tool for testing document uploads,
language routing, handwriting recognition, and inspecting layout bounding boxes.
"""

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from PIL import Image

from aqivon_omniocr import (
    IntelligentRouter,
    OmniOCRConfig,
    DEFAULT_CONFIG,
    GLOBAL_VRAM_MANAGER
)
from aqivon_omniocr.utils.visualizer import LayoutVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqivon_omniocr.app")

# Initialize global router instance
router = IntelligentRouter()


def run_ocr_interface(
    file_obj,
    language_choice,
    engine_choice,
    is_handwritten
):
    """Callback function for Gradio testing interface."""
    if file_obj is None:
        return (
            None,
            "**No document provided.** Please upload an image or PDF.",
            "",
            {},
            "Status: Awaiting input"
        )

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    doc_name = Path(file_path).name

    start_time = time.time()
    try:
        # Route and process
        result = router.route_and_process(
            document_source=file_path,
            requested_engine=None if engine_choice == "auto" else engine_choice,
            language=language_choice,
            is_handwritten=is_handwritten,
            document_name=doc_name
        )
        total_time = time.time() - start_time

        # Render first page visualization with bounding boxes
        annotated_img = None
        if result.pages:
            p1 = result.pages[0]
            # Load page 1 raw image
            raw_pages = router.preprocessor.load_document(file_path)
            if raw_pages:
                annotated_img = LayoutVisualizer.draw_page_result(
                    raw_pages[0],
                    p1,
                    show_reading_order=True,
                    show_confidence=True
                )

        # Telemetry info
        vram_stats = GLOBAL_VRAM_MANAGER.get_vram_stats()
        device_str = GLOBAL_VRAM_MANAGER.get_device()
        telemetry = (
            f"**Engine Used:** `{result.engine_used}`  |  "
            f"**Pages Processed:** `{result.num_pages}`  |  "
            f"**Total Latency:** `{result.total_latency_ms:.1f} ms`  |  "
            f"**Device:** `{device_str}`  |  "
            f"**VRAM:** `{vram_stats.get('allocated_mb', 0)} MB / {vram_stats.get('total_mb', 0)} MB`"
        )

        return (
            annotated_img,
            result.markdown,
            result.full_text,
            result.to_dict(),
            telemetry
        )

    except Exception as e:
        logger.exception("Error running OCR in interface: %s", e)
        return (
            None,
            f"### Error During Processing\n`{str(e)}`",
            "",
            {"error": str(e)},
            f"Status: Failed ({e})"
        )


def build_interface() -> gr.Blocks:
    """Constructs the lightweight testing UI."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="AQIVON OmniOCR | Testing Console", theme=theme) as demo:
        gr.Markdown(
            """
            # AQIVON OmniOCR — Diagnostic & Testing Interface
            *A unified, production-ready AI OCR platform by AQIVON Technologies.*  
            *Modular engine integration: Baidu Unlimited-OCR, PaddleOCR, TrOCR, IndicOCR, docTR, and Surya.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload Document (Image or PDF)",
                    file_types=[".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"],
                    type="filepath"
                )

                lang_dropdown = gr.Dropdown(
                    label="Target Language",
                    choices=[
                        ("Auto-Detect Language", "auto"),
                        ("English", "en"),
                        ("Hindi (हिन्दी)", "hi"),
                        ("Marathi (मराठी)", "mr"),
                        ("Urdu (اردو)", "ur"),
                        ("Bengali (বাংলা)", "bn"),
                        ("Tamil (தமிழ்)", "ta"),
                        ("Telugu (తెలుగు)", "te"),
                        ("Gujarati (ગુજરાતી)", "gu"),
                        ("Chinese (Simplified)", "zh"),
                        ("Arabic (العربية)", "ar"),
                        ("Russian (Русский)", "ru"),
                        ("German (Deutsch)", "de"),
                        ("French (Français)", "fr"),
                        ("Spanish (Español)", "es"),
                    ],
                    value="auto"
                )

                engine_dropdown = gr.Dropdown(
                    label="OCR Engine Adapter",
                    choices=[
                        ("Intelligent Router (Auto-Select)", "auto"),
                        ("AQIVON Hybrid Ensemble", "hybrid"),
                        ("Baidu Unlimited-OCR (R-SWA)", "unlimited_ocr"),
                        ("PaddleOCR Multilingual", "paddleocr"),
                        ("IndicOCR (Indian Languages)", "indicocr"),
                        ("TrOCR Handwritten Specialist", "trocr_handwritten"),
                        ("TrOCR Printed Specialist", "trocr_printed"),
                        ("Surya Layout & Line OCR", "surya"),
                        ("docTR End-to-End", "doctr"),
                        ("Tesseract Offline CPU", "tesseract"),
                    ],
                    value="auto"
                )

                hw_checkbox = gr.Checkbox(
                    label="Document Contains Handwriting",
                    value=False,
                    info="Routes to Microsoft TrOCR VisionEncoderDecoder for handwriting recognition."
                )

                run_btn = gr.Button("Run OCR", variant="primary", size="lg")
                telemetry_output = gr.Markdown("Status: Ready for testing")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Layout & Bounding Boxes"):
                        annotated_output = gr.Image(label="Annotated Page Layout", type="numpy")
                    with gr.TabItem("Structured Markdown"):
                        markdown_output = gr.Markdown(label="Generated Markdown")
                    with gr.TabItem("Extracted Plain Text"):
                        text_output = gr.TextArea(label="Plain Text Output", lines=15)
                    with gr.TabItem("JSON Data"):
                        json_output = gr.JSON(label="Structured JSON Response")

        # Hook event
        run_btn.click(
            fn=run_ocr_interface,
            inputs=[input_file, lang_dropdown, engine_dropdown, hw_checkbox],
            outputs=[annotated_output, markdown_output, text_output, json_output, telemetry_output]
        )

        gr.Markdown(
            """
            ---
            **AQIVON Technologies** • Built for NVIDIA GeForce RTX 3060 12GB • Production-ready Python Engine
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    # Launch in local testing mode
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
