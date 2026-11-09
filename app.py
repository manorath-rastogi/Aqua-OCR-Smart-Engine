"""
Aqua OCR Smart Engine — Advanced Diagnostic & Testing Interface.
A streamlined, high-performance Gradio interface for:
- Document testing (PNG, JPG, TIFF, Multi-page PDF)
- Multi-language verification (English, Hindi, Marathi, Urdu, Bengali, Chinese, Arabic, etc.)
- Visual layout & bounding box inspection with reading-order badges
- Interactive multi-page navigation, table extraction, and export (MD, TXT, JSON)
"""

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from aqua_ocr_smart_engine import (
    IntelligentRouter,
    AquaOCRConfig,
    DEFAULT_CONFIG,
    GLOBAL_VRAM_MANAGER,
    GLOBAL_PREPROCESSOR,
)
from aqua_ocr_smart_engine.types import OCRResult, PageResult
from aqua_ocr_smart_engine.utils.visualizer import LayoutVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqua_ocr_smart_engine.app")

# Shared router instance
ROUTER = IntelligentRouter()

SAMPLES_DIR = PROJECT_ROOT / "samples"


def process_document_ui(
    file_path: Optional[str],
    language_choice: str,
    engine_choice: str,
    is_handwritten: bool,
    show_reading_order: bool,
    show_confidence: bool,
    min_confidence: float
):
    """Executes OCR on uploaded file and prepares rich interactive views."""
    if not file_path or not os.path.exists(file_path):
        return (
            None,                           # annotated_image
            "**No document provided.** Please upload an image or PDF.",  # markdown_preview
            "",                             # plain_text
            pd.DataFrame(),                 # blocks_dataframe
            pd.DataFrame(),                 # tables_dataframe
            {},                             # json_output
            "Status: Awaiting input",       # telemetry_badge
            gr.update(visible=False),       # page_slider
            gr.update(value=1),             # page_slider value
            None,                           # download_md
            None,                           # download_txt
            None,                           # download_json
            None,                           # state_ocr_result
            None                            # state_raw_pages
        )

    doc_name = Path(file_path).name
    start_time = time.time()

    try:
        # Load raw images for multi-page rendering
        raw_pages = ROUTER.preprocessor.load_document(file_path)

        # Execute OCR via router
        req_engine = None if engine_choice == "auto" else engine_choice
        ocr_result: OCRResult = ROUTER.route_and_process(
            document_source=file_path,
            requested_engine=req_engine,
            language=language_choice,
            is_handwritten=is_handwritten,
            document_name=doc_name
        )

        total_latency = (time.time() - start_time) * 1000.0

        # Render first page visualization
        page_1 = ocr_result.pages[0] if ocr_result.pages else None
        annotated_img = None
        if page_1 and raw_pages:
            annotated_img = LayoutVisualizer.draw_page_result(
                raw_pages[0],
                page_1,
                show_reading_order=show_reading_order,
                show_confidence=show_confidence,
                min_confidence=min_confidence
            )

        # Build blocks dataframe for page 1
        blocks_df = _build_blocks_dataframe(page_1, min_confidence)

        # Build tables dataframe
        tables_df = _build_tables_dataframe(page_1)

        # Generate temporary files for download
        temp_dir = tempfile.gettempdir()
        base_stem = Path(doc_name).stem

        md_path = os.path.join(temp_dir, f"{base_stem}_extracted.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.markdown)

        txt_path = os.path.join(temp_dir, f"{base_stem}_extracted.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.full_text)

        json_path = os.path.join(temp_dir, f"{base_stem}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.to_json())

        # Telemetry
        vram = GLOBAL_VRAM_MANAGER.get_vram_stats()
        device_name = GLOBAL_VRAM_MANAGER.get_device()
        telemetry = (
            f"**Engine:** `{ocr_result.engine_used}` &nbsp;|&nbsp; "
            f"**Pages:** `{ocr_result.num_pages}` &nbsp;|&nbsp; "
            f"**Latency:** `{total_latency:.1f} ms` &nbsp;|&nbsp; "
            f"**Device:** `{device_name}` &nbsp;|&nbsp; "
            f"**Blocks:** `{sum(len(p.blocks) for p in ocr_result.pages)}`"
        )

        # Multi-page slider update
        slider_update = gr.update(
            visible=(ocr_result.num_pages > 1),
            minimum=1,
            maximum=max(1, ocr_result.num_pages),
            value=1,
            step=1,
            label=f"Viewing Page (1 of {ocr_result.num_pages})"
        )

        # Store serializable result dict in state
        result_dict = ocr_result.to_dict()

        return (
            annotated_img,
            ocr_result.markdown,
            ocr_result.full_text,
            blocks_df,
            tables_df,
            result_dict,
            telemetry,
            slider_update,
            1,
            md_path,
            txt_path,
            json_path,
            result_dict,
            raw_pages
        )

    except Exception as e:
        logger.exception("OCR execution failed: %s", e)
        error_msg = f"### Error During Processing\n`{str(e)}`"
        return (
            None,
            error_msg,
            str(e),
            pd.DataFrame(),
            pd.DataFrame(),
            {"error": str(e)},
            f"**Status:** Failed ({e})",
            gr.update(visible=False),
            1,
            None,
            None,
            None,
            None,
            None
        )


def on_page_change(
    page_number: int,
    state_result_dict: Optional[Dict[str, Any]],
    state_raw_pages: Optional[List[np.ndarray]],
    show_reading_order: bool,
    show_confidence: bool,
    min_confidence: float
):
    """Updates the visualizer and table inspection when switching pages without re-running OCR."""
    if not state_result_dict or not state_raw_pages:
        return None, pd.DataFrame(), pd.DataFrame()

    page_idx = int(page_number) - 1
    if page_idx < 0 or page_idx >= len(state_raw_pages):
        return None, pd.DataFrame(), pd.DataFrame()

    raw_page = state_raw_pages[page_idx]
    pages_data = state_result_dict.get("pages", [])
    if page_idx >= len(pages_data):
        return None, pd.DataFrame(), pd.DataFrame()

    # Reconstruct temporary PageResult object
    page_dict = pages_data[page_idx]
    from aqua_ocr_smart_engine.types import TextBlock, BoundingBox, BlockType, PageResult, TableResult
    
    blocks = []
    for b in page_dict.get("blocks", []):
        bb = b["bbox"]
        bbox = BoundingBox(xmin=bb["xmin"], ymin=bb["ymin"], xmax=bb["xmax"], ymax=bb["ymax"], polygon=bb.get("polygon"))
        blocks.append(TextBlock(
            text=b["text"],
            bbox=bbox,
            confidence=b.get("confidence", 1.0),
            block_type=BlockType(b.get("block_type", "paragraph")),
            reading_order_idx=b.get("reading_order_idx", -1),
            is_handwritten=b.get("is_handwritten", False)
        ))

    p_res = PageResult(
        page_number=page_dict["page_number"],
        width=page_dict["width"],
        height=page_dict["height"],
        blocks=blocks
    )

    annotated = LayoutVisualizer.draw_page_result(
        raw_page,
        p_res,
        show_reading_order=show_reading_order,
        show_confidence=show_confidence,
        min_confidence=min_confidence
    )

    blocks_df = _build_blocks_dataframe(p_res, min_confidence)
    tables_df = _build_tables_dataframe(p_res)

    return annotated, blocks_df, tables_df


def _build_blocks_dataframe(page_result: Optional[PageResult], min_conf: float = 0.0) -> pd.DataFrame:
    """Builds a Pandas DataFrame representing detected text blocks."""
    if not page_result or not page_result.blocks:
        return pd.DataFrame(columns=["#", "Text", "Confidence", "Type", "BBox [x1, y1, x2, y2]"])

    rows = []
    for b in page_result.blocks:
        if b.confidence < min_conf:
            continue
        rows.append({
            "#": b.reading_order_idx + 1 if b.reading_order_idx >= 0 else "-",
            "Text": b.text,
            "Confidence": f"{int(b.confidence * 100)}%",
            "Type": b.block_type.value.replace("_", " ").title(),
            "BBox [x1, y1, x2, y2]": f"[{int(b.bbox.xmin)}, {int(b.bbox.ymin)}, {int(b.bbox.xmax)}, {int(b.bbox.ymax)}]"
        })
    return pd.DataFrame(rows)


def _build_tables_dataframe(page_result: Optional[PageResult]) -> pd.DataFrame:
    """Builds a Pandas DataFrame representing extracted table cells if present."""
    if not page_result or not page_result.tables:
        return pd.DataFrame(columns=["Notice"], data=[["No tabular structures detected on this page."]])

    table = page_result.tables[0]
    if not table.cells:
        return pd.DataFrame(columns=["Notice"], data=[["Table present without parsed cells."]])

    # Reconstruct 2D grid
    grid = [["" for _ in range(table.num_cols)] for _ in range(table.num_rows)]
    for cell in table.cells:
        r = min(cell.row_idx, table.num_rows - 1)
        c = min(cell.col_idx, table.num_cols - 1)
        grid[r][c] = cell.text

    if len(grid) > 1:
        headers = grid[0]
        data = grid[1:]
        # Ensure unique headers
        clean_headers = [h if h.strip() else f"Col_{i+1}" for i, h in enumerate(headers)]
        return pd.DataFrame(data=data, columns=clean_headers)
    return pd.DataFrame(data=grid)


def build_app() -> gr.Blocks:
    """Constructs the AQIVON OmniOCR testing application."""
    custom_css = """
    .hero-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .badge-bar {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px;
        margin-bottom: 12px;
        font-size: 0.92rem;
    }
    .legend-pill {
        display: inline-block;
        padding: 2px 8px;
        margin-right: 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate"
    )

    with gr.Blocks(title="Aqua OCR Smart Engine Testing Console") as demo:
        # Header Banner
        gr.Markdown(
            """
            # Aqua OCR Smart Engine — Unified AI Testing Console
            **AQIVON Technologies** • High-Accuracy Multilingual & Document Layout Engine
            *Supports Baidu Unlimited-OCR (R-SWA), PaddleOCR, Microsoft TrOCR Handwriting, IndicOCR (Hindi, Marathi, Urdu, Bengali), and Multi-Page PDFs.*
            """
        )

        # State storage for multi-page document caching
        state_result_dict = gr.State(None)
        state_raw_pages = gr.State(None)

        with gr.Row():
            # Left Panel: Controls & Inputs
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### 1. Document Input")
                    input_file = gr.File(
                        label="Upload Document (Image or Multi-page PDF)",
                        file_types=[".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"],
                        type="filepath"
                    )

                with gr.Group():
                    gr.Markdown("### 2. Engine & Language Configuration")
                    lang_select = gr.Dropdown(
                        label="Target Language",
                        choices=[
                            ("Auto-Detect Language", "auto"),
                            ("English", "en"),
                            ("Hindi (हिन्दी)", "hi"),
                            ("Marathi (मराठी)", "mr"),
                            ("Urdu (اردو - RTL)", "ur"),
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

                    engine_select = gr.Dropdown(
                        label="OCR Engine Adapter",
                        choices=[
                            ("Intelligent Router (Auto-Select)", "auto"),
                            ("Aqua Hybrid Ensemble (Peak Accuracy)", "hybrid"),
                            ("Baidu Unlimited-OCR (R-SWA Layout)", "unlimited_ocr"),
                            ("PaddleOCR Multilingual (DBNet+SVTR)", "paddleocr"),
                            ("IndicOCR (Specialized Indian Scripts)", "indicocr"),
                            ("TrOCR Handwritten Specialist (Microsoft)", "trocr_handwritten"),
                            ("Surya Layout & Line OCR", "surya"),
                            ("docTR End-to-End", "doctr"),
                            ("Tesseract Offline CPU Fallback", "tesseract"),
                        ],
                        value="auto"
                    )

                    hw_check = gr.Checkbox(
                        label="Prioritize Handwriting Recognition (TrOCR)",
                        value=False,
                        info="Activates Microsoft TrOCR transformer line recognition."
                    )

                with gr.Accordion("Visualizer & Detection Tuning", open=False):
                    show_order_check = gr.Checkbox(label="Show Reading Order Badges (#1, #2...)", value=True)
                    show_conf_check = gr.Checkbox(label="Show Confidence Scores (%)", value=False)
                    conf_slider = gr.Slider(label="Min Confidence Threshold", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

                run_btn = gr.Button("⚡ Run Aqua OCR Smart Engine", variant="primary", size="lg")

                # Sample Documents Selector
                sample_files = [
                    str(SAMPLES_DIR / "sample_english_invoice.png"),
                    str(SAMPLES_DIR / "sample_indic_hindi.png"),
                    str(SAMPLES_DIR / "sample_handwriting.png"),
                    str(SAMPLES_DIR / "sample_multipage_document.pdf")
                ]
                existing_samples = [s for s in sample_files if os.path.exists(s)]
                if existing_samples:
                    gr.Markdown("#### Quick Sample Documents:")
                    gr.Examples(
                        examples=[[s] for s in existing_samples],
                        inputs=[input_file],
                        label="Click any sample document to test instantly:"
                    )

            # Right Panel: Results & Telemetry
            with gr.Column(scale=7):
                telemetry_badge = gr.Markdown("⚡ **Status:** Ready for document input")

                # Multi-Page Navigation Slider (Appears when PDF > 1 page)
                page_slider = gr.Slider(
                    label="Page Navigation",
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    visible=False
                )

                with gr.Tabs() as result_tabs:
                    with gr.TabItem("🖼️ Layout & Bounding Boxes"):
                        gr.Markdown(
                            """
                            **Legend:** 
                            <span class="legend-pill" style="background:#9c27b0">Purple: Headings</span>
                            <span class="legend-pill" style="background:#2196f3">Blue: Paragraphs</span>
                            <span class="legend-pill" style="background:#ff7043">Orange: Handwriting</span>
                            <span class="legend-pill" style="background:#4caf50">Green: Tables</span>
                            <span class="legend-pill" style="background:#ffb300">Yellow: Formulas</span>
                            """
                        )
                        annotated_view = gr.Image(label="Annotated Page Layout", type="numpy")

                    with gr.TabItem("📝 Structured Markdown"):
                        markdown_view = gr.Markdown(label="Rendered Markdown Output")

                    with gr.TabItem("📄 Plain Extracted Text"):
                        text_view = gr.Textbox(label="Plain Text Output", lines=18)

                    with gr.TabItem("📊 Extracted Tables"):
                        tables_view = gr.DataFrame(label="Detected Tables", wrap=True)

                    with gr.TabItem("🔍 Text Blocks & BBoxes Table"):
                        blocks_view = gr.DataFrame(label="Detected Text Blocks", wrap=True)

                    with gr.TabItem("⚙️ JSON & Telemetry"):
                        json_view = gr.JSON(label="Full Structured JSON Schema")

                # Export / Download Buttons
                gr.Markdown("#### Export Extracted Results:")
                with gr.Row():
                    dl_md = gr.File(label="Download Markdown (.md)", interactive=False)
                    dl_txt = gr.File(label="Download Text (.txt)", interactive=False)
                    dl_json = gr.File(label="Download JSON (.json)", interactive=False)

        # Event: Run OCR
        run_btn.click(
            fn=process_document_ui,
            inputs=[
                input_file,
                lang_select,
                engine_select,
                hw_check,
                show_order_check,
                show_conf_check,
                conf_slider
            ],
            outputs=[
                annotated_view,
                markdown_view,
                text_view,
                blocks_view,
                tables_view,
                json_view,
                telemetry_badge,
                page_slider,
                page_slider,
                dl_md,
                dl_txt,
                dl_json,
                state_result_dict,
                state_raw_pages
            ]
        )

        # Event: Page Slider Change (Instant re-render without re-running OCR)
        page_slider.change(
            fn=on_page_change,
            inputs=[
                page_slider,
                state_result_dict,
                state_raw_pages,
                show_order_check,
                show_conf_check,
                conf_slider
            ],
            outputs=[annotated_view, blocks_view, tables_view]
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Aqua OCR Smart Engine** • AQIVON Technologies • GPU Acceleration for RTX 3060 12GB • [NOTICES.md](NOTICES.md)
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    print("Launching Aqua OCR Smart Engine Testing Interface on http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
