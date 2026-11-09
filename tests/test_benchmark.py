"""
End-to-end benchmark and accuracy verification for Aqua OCR Smart Engine.
Tests PaddleOCR, IndicOCR, TrOCR, and Hybrid engines on synthetic document samples.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqua_ocr_smart_engine.router import IntelligentRouter
from aqua_ocr_smart_engine.utils.metrics import OCRMetrics


def run_benchmark():
    print("=" * 60)
    print("Aqua OCR Smart Engine System Benchmark & Verification")
    print("=" * 60)

    router = IntelligentRouter()
    samples_dir = PROJECT_ROOT / "samples"
    
    test_cases = [
        {
            "name": "English Business Invoice with Table",
            "file": samples_dir / "sample_english_invoice.png",
            "engine": "hybrid",
            "language": "en",
            "is_handwritten": False,
            "expected_keywords": ["AQIVON", "INVOICE", "Item Description", "$3,500", "Total Due"]
        },
        {
            "name": "Indic Language (Hindi / Devanagari) Document",
            "file": samples_dir / "sample_indic_hindi.png",
            "engine": "indicocr",
            "language": "hi",
            "is_handwritten": False,
            "expected_keywords": ["अकिवोन", "ओसीआर", "भारतीय", "हिंदी"]
        },
        {
            "name": "Handwritten Prescription Note",
            "file": samples_dir / "sample_handwriting.png",
            "engine": "hybrid",
            "language": "en",
            "is_handwritten": True,
            "expected_keywords": ["Prescription", "Paracetamol", "water", "Monday"]
        }
    ]

    total_pages = 0
    total_time = 0.0

    for case in test_cases:
        path = case["file"]
        if not path.exists():
            print(f"[SKIP] Sample file not found: {path}")
            continue

        print(f"\n---> Testing: {case['name']}")
        print(f"     Engine: {case['engine']}, Language: {case['language']}")

        start = time.time()
        result = router.route_and_process(
            document_source=str(path),
            requested_engine=case["engine"],
            language=case["language"],
            is_handwritten=case["is_handwritten"],
            document_name=path.name
        )
        elapsed = time.time() - start
        total_time += elapsed
        total_pages += result.num_pages

        print(f"     Status: Success! Pages: {result.num_pages}, Latency: {result.total_latency_ms:.1f}ms")
        print(f"     Blocks Detected: {sum(len(p.blocks) for p in result.pages)}")
        print(f"     Engine Used: {result.engine_used}")

        # Check expected keywords
        extracted_text = result.full_text
        found_keywords = [kw for kw in case["expected_keywords"] if kw.lower() in extracted_text.lower()]
        keyword_recall = (len(found_keywords) / len(case["expected_keywords"])) * 100.0

        print(f"     Keyword Recall: {keyword_recall:.1f}% ({len(found_keywords)}/{len(case['expected_keywords'])})")
        print("     Preview Extracted Text:")
        for line in extracted_text.split("\n")[:4]:
            if line.strip():
                print(f"       | {line.strip()}")

        if result.markdown:
            print("     Markdown Sample:")
            for line in result.markdown.split("\n")[:4]:
                if line.strip():
                    print(f"       # {line.strip()}")

    print("\n" + "=" * 60)
    print(f"Benchmark Summary: Processed {total_pages} pages in {total_time:.2f}s ({total_pages / max(0.001, total_time):.2f} pages/sec)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

