"""
Generates synthetic sample documents for benchmarking AQIVON OmniOCR:
1. English Business Invoice / Document with Table
2. Hindi (हिन्दी) & Marathi (मराठी) Devanagari text
3. Simulated Handwritten Note
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = Path(__file__).resolve().parent
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def get_font(font_name: str, size: int) -> ImageFont.ImageFont:
    """Attempts to load a Windows TrueType font, falling back to default."""
    win_font = Path("C:/Windows/Fonts") / font_name
    if win_font.exists():
        try:
            return ImageFont.truetype(str(win_font), size)
        except Exception:
            pass
    return ImageFont.load_default()


def generate_english_document():
    img = Image.new("RGB", (900, 1100), color="white")
    draw = ImageDraw.Draw(img)

    title_font = get_font("arialbd.ttf", 32)
    heading_font = get_font("arialbd.ttf", 22)
    body_font = get_font("arial.ttf", 18)

    # Title
    draw.text((50, 40), "AQIVON TECHNOLOGIES - INVOICE", fill="black", font=title_font)
    draw.line((50, 85, 850, 85), fill="black", width=2)

    # Metadata
    draw.text((50, 100), "Customer: Global Logistics Corp", fill="black", font=body_font)
    draw.text((50, 130), "Invoice ID: AQV-2026-9042", fill="black", font=body_font)
    draw.text((50, 160), "Date: November 08, 2026", fill="black", font=body_font)

    # Section Heading
    draw.text((50, 210), "Service Summary and Line Items", fill="black", font=heading_font)

    # Table Header Box
    draw.rectangle([50, 250, 850, 290], outline="black", width=2, fill="#e6e6e6")
    draw.text((70, 260), "Item Description", fill="black", font=heading_font)
    draw.text((450, 260), "Qty", fill="black", font=heading_font)
    draw.text((580, 260), "Rate", fill="black", font=heading_font)
    draw.text((720, 260), "Amount", fill="black", font=heading_font)

    # Table Row 1
    draw.rectangle([50, 290, 850, 340], outline="black", width=1)
    draw.text((70, 305), "Aqua OCR Smart Engine License", fill="black", font=body_font)
    draw.text((460, 305), "1", fill="black", font=body_font)
    draw.text((580, 305), "$3,500", fill="black", font=body_font)
    draw.text((720, 305), "$3,500", fill="black", font=body_font)

    # Table Row 2
    draw.rectangle([50, 340, 850, 390], outline="black", width=1)
    draw.text((70, 355), "GPU Cluster Setup & RTX Optimization", fill="black", font=body_font)
    draw.text((460, 355), "2", fill="black", font=body_font)
    draw.text((580, 355), "$750", fill="black", font=body_font)
    draw.text((720, 355), "$1,500", fill="black", font=body_font)

    # Total Box
    draw.rectangle([50, 390, 850, 440], outline="black", width=2, fill="#f2f2f2")
    draw.text((550, 405), "Total Due:", fill="black", font=heading_font)
    draw.text((720, 405), "$5,000", fill="black", font=heading_font)

    # Note
    draw.text((50, 480), "Thank you for your partnership with AQIVON Technologies.", fill="black", font=body_font)

    out_path = SAMPLES_DIR / "sample_english_invoice.png"
    img.save(out_path)
    print(f"Saved: {out_path}")


def generate_indic_document():
    img = Image.new("RGB", (900, 800), color="white")
    draw = ImageDraw.Draw(img)

    # Nirmala UI font supports Hindi, Marathi, Bengali, Tamil, etc. on Windows
    title_font = get_font("NirmalaB.ttf", 28)
    body_font = get_font("Nirmala.ttf", 20)

    # Hindi Heading
    draw.text((50, 50), "अकिवोन टेक्नोलॉजीज - स्मार्ट ओसीआर इंजन", fill="black", font=title_font)
    draw.line((50, 95, 850, 95), fill="black", width=2)

    # Hindi & Marathi Paragraphs
    hindi_p1 = "अकिवोन ओम्नी ओसीआर एक उच्च सटीकता वाला भारतीय भाषा ओसीआर इंजन है।"
    hindi_p2 = "यह हिंदी, मराठी, बंगाली और उर्दू सहित सभी भारतीय भाषाओं का समर्थन करता है।"
    marathi_p = "मराठी दस्तऐवज आणि हस्तलिखित मजकूर ओळखण्यासाठी अत्यंत अचूक आणि वेगवान प्रणाली."

    draw.text((50, 130), hindi_p1, fill="black", font=body_font)
    draw.text((50, 180), hindi_p2, fill="black", font=body_font)
    draw.text((50, 240), marathi_p, fill="black", font=body_font)

    # Multilingual List
    draw.text((50, 320), "प्रमुख विशेषताएं (Key Features):", fill="black", font=title_font)
    draw.text((70, 370), "1. उच्च गति और सटीकता (High Speed & Precision)", fill="black", font=body_font)
    draw.text((70, 410), "2. बहुभाषी दस्तावेज़ प्रसंस्करण (Multilingual Support)", fill="black", font=body_font)
    draw.text((70, 450), "3. स्वचालित तालिका और लेआउट पुनर्प्राप्ति (Tables & Layout)", fill="black", font=body_font)

    out_path = SAMPLES_DIR / "sample_indic_hindi.png"
    img.save(out_path)
    print(f"Saved: {out_path}")


def generate_handwritten_document():
    img = Image.new("RGB", (800, 500), color="#fffdf8")
    draw = ImageDraw.Draw(img)

    # Try Lucida Handwriting or Comic or Segoe Print
    hw_font = get_font("segprb.ttf", 24)
    if hw_font == ImageFont.load_default():
        hw_font = get_font("comic.ttf", 24)

    # Simulated handwriting text
    draw.text((60, 50), "Patient Prescription Notes:", fill="#1a2035", font=hw_font)
    draw.text((60, 110), "Rx: Paracetamol 500mg twice daily after meals.", fill="#1a2035", font=hw_font)
    draw.text((60, 170), "Drink plenty of water and rest for 3 days.", fill="#1a2035", font=hw_font)
    draw.text((60, 240), "Follow up on Monday morning at 10 AM.", fill="#1a2035", font=hw_font)
    draw.text((450, 340), "Dr. A. Sharma, MD", fill="#1a2035", font=hw_font)

    out_path = SAMPLES_DIR / "sample_handwriting.png"
    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_english_document()
    generate_indic_document()
    generate_handwritten_document()

