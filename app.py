"""
Aqua OCR Smart Engine — Production AI Document Layout & OCR Platform.
Preserves original document visual layouts:
- Multi-column resumes, research papers, reports, invoices
- Reading order flow (Headers -> Left Column -> Right Column -> Footers)
- Headings, section titles, bullet lists, and tables
- Side-by-side visual comparison between original scan and reconstructed layout
- Multi-engine flexibility: Auto, Surya (Layout & OCR), PaddleOCR, IndicOCR, Tesseract
"""

import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paddle environment flags for Windows CPU/GPU stability
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import gradio as gr
from aqua_ocr_smart_engine import IntelligentRouter
from aqua_ocr_smart_engine.types import OCRResult
from aqua_ocr_smart_engine.utils.visualizer import LayoutVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqua_ocr_smart_engine.app")

# Initialize global intelligent router
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
    "Surya (Layout & OCR)": "surya",
    "PaddleOCR": "paddleocr",
    "IndicOCR": "indicocr",
    "Tesseract": "tesseract",
}


def load_file_preview(file_path: Optional[str]) -> Optional[np.ndarray]:
    """Loads first page image of uploaded file for instant display."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        pages = ROUTER.preprocessor.load_document(file_path)
        return pages[0] if pages else None
    except Exception as e:
        logger.warning("Could not load preview for %s: %s", file_path, e)
        return None


def run_ocr_pipeline(
    file_path: Optional[str],
    language_display: str,
    engine_display: str,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray], str, Dict[str, Any], str, str]:
    """
    Executes full layout detection and OCR pipeline.
    Returns:
        (original_preview, html_preview, annotated_image, markdown, json_data, plain_text, status_msg)
    """
    if not file_path or not os.path.exists(file_path):
        empty_html = "<div style='padding:20px; color:#888;'>Please upload a document to begin.</div>"
        return (
            None,
            empty_html,
            None,
            "",
            {},
            "",
            "⚠️ Error: Please upload a valid document or image before running OCR."
        )

    lang_code = LANGUAGE_MAP.get(language_display, "auto")
    engine_id = ENGINE_MAP.get(engine_display, None)
    file_name = Path(file_path).name

    start_time = time.time()
    try:
        progress(0.1, desc="Loading document & preprocessing...")
        raw_pages = ROUTER.preprocessor.load_document(file_path)
        original_img = raw_pages[0] if raw_pages else None

        progress(0.3, desc=f"Analyzing layout with {engine_display}...")
        ocr_result: OCRResult = ROUTER.route_and_process(
            document_source=file_path,
            requested_engine=engine_id,
            language=lang_code,
            document_name=file_name
        )

        progress(0.8, desc="Reconstructing document layout & columns...")
        elapsed_sec = time.time() - start_time

        # 1. Annotated image visualization with reading order badges
        annotated_img = None
        first_page = ocr_result.pages[0] if ocr_result.pages else None
        if original_img is not None and first_page is not None:
            annotated_img = LayoutVisualizer.draw_page_result(
                original_img,
                first_page,
                show_reading_order=True,
                show_confidence=False
            )

        # 2. Formatted Layout HTML
        html_content = ocr_result.html_preview
        if not html_content and first_page:
            from aqua_ocr_smart_engine.postprocessor.html_reconstructor import HTMLDocumentReconstructor
            html_content = HTMLDocumentReconstructor.render_page(first_page)

        # 3. Clean Text in Natural Reading Order
        if len(ocr_result.pages) == 1:
            clean_text = ocr_result.pages[0].text.strip()
        else:
            clean_text = ocr_result.full_text.strip()

        # 4. Structured JSON representation
        json_data = ocr_result.to_dict()

        # 5. Status Diagnostics
        engine_used_name = ocr_result.metadata.get("engine_name") or ocr_result.engine_used
        total_blocks = sum(len(p.blocks) for p in ocr_result.pages)
        device = ocr_result.metadata.get("device", "CPU")

        status_msg = (
            f"✅ Layout & OCR Completed in {elapsed_sec:.2f}s | "
            f"Engine: {engine_used_name} | "
            f"Pages: {ocr_result.num_pages} | "
            f"Blocks: {total_blocks} | "
            f"Device: {device}"
        )
        if engine_display == "Tesseract" and ocr_result.engine_used != "tesseract":
            status_msg += " (Note: Tesseract not found on system PATH; processed via PaddleOCR)"

        progress(1.0, desc="Completed!")
        return (
            original_img,
            html_content,
            annotated_img,
            ocr_result.markdown,
            json_data,
            clean_text,
            status_msg
        )

    except Exception as exc:
        logger.exception("OCR Pipeline error on '%s': %s", file_name, exc)
        err_html = f"<div style='padding:20px; color:#d32f2f;'>Error processing document: {html_escape(str(exc))}</div>"
        return (
            None,
            err_html,
            None,
            "",
            {"error": str(exc)},
            "",
            f"❌ Execution Error: {type(exc).__name__} - {str(exc)}"
        )


def html_escape(text: str) -> str:
    import html
    return html.escape(text)


def build_interface() -> gr.Blocks:
    """Creates a side-by-side layout-preserving Gradio UI for Aqua OCR Smart Engine."""
    with gr.Blocks(title="Aqua OCR Smart Engine") as demo:
        gr.Markdown(
            """
            # Aqua OCR Smart Engine
            **Unified Document Layout Analysis & AI OCR Platform** &bull; AQIVON Technologies
            *Preserves original multi-column layouts, reading order, headings, and formatting side-by-side with original scans.*
            """
        )

        with gr.Row():
            # Left Column: Upload, Controls, & Original Document
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload Document / PDF / Image",
                    file_types=["image", ".pdf"],
                    type="filepath"
                )

                original_preview = gr.Image(
                    label="Original Document Scan",
                    interactive=False
                )

                with gr.Row():
                    language_selector = gr.Dropdown(
                        label="Language",
                        choices=list(LANGUAGE_MAP.keys()),
                        value="Auto",
                        interactive=True,
                        scale=1
                    )
                    engine_selector = gr.Dropdown(
                        label="OCR & Layout Engine",
                        choices=list(ENGINE_MAP.keys()),
                        value="Auto",
                        interactive=True,
                        scale=1
                    )

                run_btn = gr.Button("Run OCR & Layout Analysis", variant="primary", size="lg")

                status_box = gr.Textbox(
                    label="Status & Execution Diagnostics",
                    value="Ready. Upload a document or select a sample to begin.",
                    lines=2,
                    interactive=False
                )

            # Right Column: Reconstructed Formatted Layout & Structured Views
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📄 Reconstructed Formatted Layout"):
                        output_html = gr.HTML(
                            value="<div style='padding:30px; text-align:center; color:#888;'>Upload a document and click 'Run OCR & Layout Analysis' to see the reconstructed layout.</div>",
                            label="Layout Faithful Document"
                        )

                    with gr.TabItem("🖼️ Visual Layout & Reading Order"):
                        output_annotated = gr.Image(
                            label="Document with Bounding Boxes & Reading Order Badges (#1, #2...)",
                            interactive=False
                        )

                    with gr.TabItem("📝 Structured Markdown"):
                        output_md = gr.Textbox(
                            label="Structured Markdown (Columns & Sections)",
                            lines=18,
                            buttons=["copy"],
                            interactive=False
                        )

                    with gr.TabItem("📊 Structured JSON"):
                        output_json = gr.JSON(
                            label="Structured JSON (Positions, Hierarchy, Confidence)"
                        )

                    with gr.TabItem("📜 Plain Text"):
                        output_text = gr.Textbox(
                            label="Natural Reading Order Text",
                            lines=18,
                            buttons=["copy"],
                            interactive=False
                        )

        # Update original image preview on file upload
        input_file.change(
            fn=load_file_preview,
            inputs=[input_file],
            outputs=[original_preview]
        )

        # Run pipeline
        run_btn.click(
            fn=run_ocr_pipeline,
            inputs=[input_file, language_selector, engine_selector],
            outputs=[
                original_preview,
                output_html,
                output_annotated,
                output_md,
                output_json,
                output_text,
                status_box
            ]
        )

        # Quick Test Samples
        samples = []
        sample_resume = SAMPLES_DIR / "sample_resume_luna.jpg"
        sample_invoice = SAMPLES_DIR / "sample_english_invoice.png"
        sample_hindi = SAMPLES_DIR / "sample_indic_hindi.png"

        if sample_resume.exists():
            samples.append([str(sample_resume), "English", "Auto"])
        if sample_invoice.exists():
            samples.append([str(sample_invoice), "English", "PaddleOCR"])
        if sample_hindi.exists():
            samples.append([str(sample_hindi), "Hindi", "IndicOCR"])

        if samples:
            gr.Examples(
                examples=samples,
                inputs=[input_file, language_selector, engine_selector],
                outputs=[
                    original_preview,
                    output_html,
                    output_annotated,
                    output_md,
                    output_json,
                    output_text,
                    status_box
                ],
                fn=run_ocr_pipeline,
                cache_examples=False,
                label="Quick Test Samples (Click to Load)"
            )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
