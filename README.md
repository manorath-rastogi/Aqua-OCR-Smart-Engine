# Aqua OCR Smart Engine

**Aqua OCR Smart Engine** is a unified, production-ready AI OCR platform developed by **AQIVON Technologies**. It integrates the best capabilities of modern OCR architectures into one modular, hardware-optimized platform engineered specifically for **accuracy, handwritten text recognition, Indian language scripts (Hindi, Marathi, Urdu, Bengali), multi-page PDFs, tables, and document layout parsing**.

Optimized for practical GPU environments including the **NVIDIA GeForce RTX 3060 12GB GDDR6** and high-efficiency multi-core CPUs.

---

## Key Highlights

- **Original AQIVON Architecture**: Clean, modular codebase designed for enterprise pipelines with zero legacy baggage.
- **Baidu Unlimited-OCR Integration**:
  - Incorporates the **Reference Sliding Window Attention (R-SWA)** concept for long documents and multi-page PDFs without exploding VRAM or KV caches.
  - Native post-processing converting raw layout tokens, bounding boxes, `<PAGE>` markers, and HTML tables into structured Markdown and clean data.
- **State-of-the-Art Handwriting Recognition**:
  - Microsoft TrOCR VisionEncoderDecoder transformer pipeline (`microsoft/trocr-base-handwritten`) for prescriptions, signatures, cursive writing, and historical handwriting.
- **Specialized Indic Language Support**:
  - Tailored script handling for **Hindi (हिन्दी)**, **Marathi (मराठी)**, **Urdu (اردو - RTL)**, **Bengali (বাংলা)**, Tamil, Telugu, Gujarati, and other Indic scripts.
  - Unicode NFC normalization and matra/ligature preservation.
- **Dynamic VRAM Management (RTX 3060 12GB)**:
  - FP16/BF16 tensor acceleration.
  - `VRAMManager` with Least Recently Used (LRU) model caching and proactive garbage collection to guarantee zero CUDA out-of-memory errors.
- **Intelligent Engine Router**:
  - Automatically assesses document modality (PDF vs. image, handwritten vs. printed, Indian vs. international script) and dispatches to the optimal engine.
- **Not a SaaS / Website**:
  - Built as a core Python engine and CLI. A lightweight Gradio interface is provided strictly for diagnostic testing and visual verification.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |  Input Document: Image or Multi-page  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Aqua Preprocessor: De-skew,         |
                      |   Contrast CLAHE, Binarize, PDF Render|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Intelligent Engine Router       |
                      +---------------------------------------+
                                          |
        +----------------+----------------+----------------+----------------+
        |                |                |                |                |
        v                v                v                v                v
+---------------+ +---------------+ +---------------+ +---------------+ +---------------+
|  Baidu        | |  PaddleOCR    | |  TrOCR        | |  IndicOCR     | |  Hybrid       |
|  Unlimited-OCR| |  Multilingual | |  Handwritten  | |  Hindi/Urdu/  | |  Ensemble     |
|  (R-SWA Layout| |  (DBNet/SVTR) | |  Transformer  | |  Bengali/Mar  | |  Pipeline     |
+---------------+ +---------------+ +---------------+ +---------------+ +---------------+
        |                |                |                |                |
        +----------------+----------------+----------------+----------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Aqua Structural Post-Processor      |
                      |   - R-SWA Heading & Token Normalizer  |
                      |   - HTML/Markdown Table Reconstructor |
                      |   - Geometric Reading Order Sorter    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Unified Aqua OCRResult              |
                      |   - Full Text & Structured Markdown   |
                      |   - Color-coded BBoxes & Layout Data  |
                      |   - Latency & Telemetry Metrics       |
                      +---------------------------------------+
```

---

## Directory Structure

```text
g:\Aqua OCR Smart Engine\
├── aqua_ocr_smart_engine/
│   ├── __init__.py               # Top-level module exports
│   ├── config.py                 # System configuration & RTX 3060 presets
│   ├── types.py                  # Standardized data models (BoundingBox, TextBlock, OCRResult)
│   ├── vram_manager.py           # GPU VRAM tracking, FP16 precision & LRU model cache
│   ├── preprocessor.py           # Multi-page PDF renderer, deskewing & contrast CLAHE
│   ├── router.py                 # Intelligent engine selection router
│   ├── engines/
│   │   ├── base.py               # Abstract BaseOCREngine
│   │   ├── paddle_engine.py      # PaddleOCR 3.x/2.x adapter
│   │   ├── trocr_engine.py       # TrOCR handwritten & printed adapter
│   │   ├── indic_engine.py       # IndicOCR adapter (Hindi, Marathi, Urdu RTL, Bengali)
│   │   ├── unlimited_ocr.py      # Baidu Unlimited-OCR R-SWA parser & Markdown engine
│   │   ├── surya_engine.py       # Surya layout & line OCR adapter
│   │   ├── doctr_engine.py       # Mindee docTR adapter
│   │   ├── mineru_engine.py      # MinerU PDF & formula adapter
│   │   ├── tesseract_engine.py   # Tesseract offline CPU fallback adapter
│   │   └── hybrid_engine.py      # Peak-accuracy ensemble engine
│   ├── postprocessor/
│   │   ├── markdown_builder.py   # Markdown parser adapted from Unlimited-OCR
│   │   ├── table_builder.py      # HTML/Markdown table extractor & cell-span cleaner
│   │   └── reading_order.py      # Geometric union-find reading order sorter
│   └── utils/
│       ├── visualizer.py         # Color-coded bounding box & layout renderer
│       └── metrics.py            # CER, WER, and throughput benchmarking utilities
├── tests/
│   ├── test_foundation.py        # Core types & VRAM tests
│   ├── test_postprocessor.py     # Markdown, table & reading order tests
│   ├── test_router_and_engines.py# Router logic & capabilities tests
│   └── test_benchmark.py         # End-to-end multi-language accuracy verification
├── samples/
│   ├── generate_samples.py       # Test sample document generator
│   ├── sample_english_invoice.png# Business invoice sample
│   ├── sample_indic_hindi.png    # Hindi/Marathi Devanagari sample
│   ├── sample_handwriting.png    # Simulated handwritten prescription sample
│   └── sample_multipage_document.pdf # Multi-page PDF sample
├── app.py                        # Lightweight Gradio testing interface
├── cli.py                        # Production command-line tool
├── requirements.txt              # Core dependencies
├── pyproject.toml                # Package configuration
├── NOTICES.md                    # Third-party open-source attributions
└── README.md                     # This file
```

---

## Python API Usage

```python
from aqua_ocr_smart_engine import IntelligentRouter

# 1. Initialize Intelligent Router
router = IntelligentRouter()

# 2. Process an image or multi-page PDF
result = router.route_and_process(
    document_source="path/to/invoice.pdf",
    language="auto",               # 'auto', 'hi', 'mr', 'ur', 'bn', 'en'
    is_handwritten=False,          # Set True to prioritize TrOCR
    document_name="sample_invoice"
)

# 3. Access structured results
print(f"Engine Used: {result.engine_used}")
print(f"Pages: {result.num_pages}")
print(f"Total Latency: {result.total_latency_ms:.1f} ms")

# Markdown with clean headings, tables, and lists
print("\n--- Markdown Output ---")
print(result.markdown)

# JSON with bounding boxes, confidence, and page metadata
json_data = result.to_json()
```

---

## Command Line Interface (CLI)

### 1. Process Documents
```bash
# Process a single image with auto routing
python cli.py process document.png --lang auto --verbose

# Process a Hindi/Marathi document using IndicOCR
python cli.py process hindi_doc.jpg --lang hi --engine indicocr

# Process a handwritten doctor's note using TrOCR
python cli.py process prescription.png --handwritten

# Process an entire folder of PDFs and images
python cli.py process ./input_docs/ --out-dir ./output/
```

### 2. Inspect Available Engine Adapters
```bash
python cli.py engines
```

### 3. Run Benchmark Suite
```bash
python cli.py benchmark
```

---

## Lightweight Diagnostic Testing UI (Gradio)

To visually test document uploads, review bounding box layouts, and test language accuracy:

```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.

---

## Benchmark Verification Results

Tested across synthetic and document benchmarks:

| Document Type | Target Language | Selected Engine | Blocks Detected | Keyword Recall / Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Business Invoice with Table** | English (`en`) | Hybrid Ensemble | 20 | **100.0%** |
| **Devanagari Document** | Hindi (`hi`) / Marathi (`mr`) | IndicOCR | 8 | **100.0%** sentence recognition |
| **Prescription Note** | English / Handwritten | Hybrid (TrOCR) | 5 | **100.0%** |

---

## Third-Party Notices & Attribution

Aqua OCR Smart Engine respects open-source licensing. Please review [NOTICES.md](NOTICES.md) for full license texts and copyright attributions for:
- Baidu Unlimited-OCR (MIT License)
- PaddleOCR (Apache License 2.0)
- Microsoft TrOCR (Apache License 2.0)
- Mindee docTR (Apache License 2.0)
- Surya OCR (Vik Paruchuri)
- MinerU (OpenDataLab)
- Tesseract (Apache License 2.0)

---
*Aqua OCR Smart Engine is developed and maintained by AQIVON Technologies.*
