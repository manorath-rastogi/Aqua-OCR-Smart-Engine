"""
AQIVON OmniOCR Command Line Interface (CLI).
Production-grade CLI for single documents, multi-page PDFs, batch directory processing,
engine listing, and automated benchmarking.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqivon_omniocr import (
    IntelligentRouter,
    OmniOCRConfig,
    DEFAULT_CONFIG,
    GLOBAL_VRAM_MANAGER
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aqivon_omniocr.cli")


def cmd_process(args):
    """Processes a single file or a directory of documents."""
    router = IntelligentRouter()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pdf"}
        files_to_process = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]

    if not files_to_process:
        logger.warning("No supported document files found to process.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / "omniocr_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAQIVON OmniOCR: Processing {len(files_to_process)} document(s)...")
    print(f"Target Language: {args.lang} | Requested Engine: {args.engine or 'Auto'} | Handwritten: {args.handwritten}")
    print("-" * 65)

    for doc_file in files_to_process:
        start = time.time()
        print(f"--> Processing: {doc_file.name} ... ", end="", flush=True)

        try:
            result = router.route_and_process(
                document_source=str(doc_file),
                requested_engine=args.engine if args.engine != "auto" else None,
                language=args.lang,
                is_handwritten=args.handwritten,
                document_name=doc_file.name
            )
            elapsed = time.time() - start
            print(f"Done in {elapsed:.2f}s ({result.num_pages} page(s), engine: {result.engine_used})")

            # Save outputs
            base_stem = doc_file.stem
            
            # 1. Plain text
            txt_file = out_dir / f"{base_stem}.txt"
            txt_file.write_text(result.full_text, encoding="utf-8")

            # 2. Markdown
            md_file = out_dir / f"{base_stem}.md"
            md_file.write_text(result.markdown, encoding="utf-8")

            # 3. JSON
            json_file = out_dir / f"{base_stem}.json"
            json_file.write_text(result.to_json(), encoding="utf-8")

            if args.verbose:
                print("\n[Preview Markdown]:")
                print(result.markdown[:500] + ("..." if len(result.markdown) > 500 else ""))
                print("-" * 65)

        except Exception as e:
            print(f"FAILED: {e}")
            logger.exception("Error processing document %s", doc_file)

    print(f"\nAll outputs saved to: {out_dir}\n")


def cmd_engines(args):
    """Lists registered OCR engine adapters and GPU hardware status."""
    router = IntelligentRouter()
    caps = router.list_engines()
    vram_stats = GLOBAL_VRAM_MANAGER.get_vram_stats()
    device = GLOBAL_VRAM_MANAGER.get_device()

    print("\n" + "=" * 65)
    print("AQIVON OmniOCR - Engine Adapters & Hardware Status")
    print("=" * 65)
    print(f"Active Device: {device}")
    print(f"VRAM Allocated: {vram_stats['allocated_mb']} MB / Total: {vram_stats['total_mb']} MB (Free: {vram_stats['free_mb']} MB)")
    print("-" * 65)
    print(f"{'Engine ID':<20} | {'Handwriting':<12} | {'Tables':<8} | {'PDF':<5} | {'Description'}")
    print("-" * 65)

    for c in caps:
        hw_str = "Yes" if c.supports_handwriting else "No"
        tbl_str = "Yes" if c.supports_tables else "No"
        pdf_str = "Yes" if c.supports_multipage_pdf else "No"
        print(f"{c.engine_id:<20} | {hw_str:<12} | {tbl_str:<8} | {pdf_str:<5} | {c.description[:45]}")

    print("=" * 65 + "\n")


def cmd_benchmark(args):
    """Runs the benchmark suite on sample documents."""
    from tests.test_benchmark import run_benchmark
    run_benchmark()


def main():
    parser = argparse.ArgumentParser(
        description="AQIVON OmniOCR: Unified AI OCR Platform by AQIVON Technologies"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Command: process
    proc_parser = subparsers.add_parser("process", help="Process an image or PDF document")
    proc_parser.add_argument("input", type=str, help="Path to document file or folder")
    proc_parser.add_argument("--lang", type=str, default="auto", help="Language code (en, hi, mr, ur, bn, auto)")
    proc_parser.add_argument("--engine", type=str, default="auto",
                             choices=["auto", "hybrid", "unlimited_ocr", "paddleocr", "indicocr", "trocr_handwritten", "trocr_printed", "surya", "doctr", "tesseract"],
                             help="OCR engine adapter to use")
    proc_parser.add_argument("--handwritten", action="store_true", help="Flag document as containing handwriting")
    proc_parser.add_argument("--out-dir", type=str, default=None, help="Directory to save output files")
    proc_parser.add_argument("--verbose", action="store_true", help="Print preview in console")

    # Command: engines
    subparsers.add_parser("engines", help="List registered engines and hardware status")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Run benchmark verification on test samples")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "engines":
        cmd_engines(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

