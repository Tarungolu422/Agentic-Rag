"""
ocr_engine.py — Phase 2: OCR & Multi-Modal Text Extraction
────────────────────────────────────────────────────────────
Extracts text from scanned PDFs, images, and charts using Tesseract OCR.
Dependencies:
- wrappers: pip install pytesseract pdf2image
- binaries: Tesseract-OCR, Poppler (must be in system PATH)
"""

import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Optional

import sys

# Configure Tesseract path if on Windows (local testing)
if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux (Streamlit Cloud), it's installed system-wide via packages.txt so no path is needed.

def check_dependencies() -> bool:
    """Checks if Tesseract and Poppler are available."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print("⚠️ Tesseract not found. Please install it and add to PATH.")
        return False

def extract_text_from_image(image: Image.Image) -> str:
    """Uses Tesseract to extract text from a PIL Image."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"❌ OCR Error on image: {e}")
        return ""

# Poppler configuration: sys.platform aware
POPPLER_PATH = r"C:\Users\Tarun\poppler-25.12.0\Library\bin" if sys.platform == "win32" else None

def ocr_pdf(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    Converts a PDF to images (one per page) and runs OCR on each.
    Returns a list of strings (one per page).
    """
    if not check_dependencies():
        return []

    print(f"📷 [OCR] Processing '{os.path.basename(pdf_path)}' with OCR (this may take a while)...")
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
        page_texts = []
        
        for i, img in enumerate(images):
            print(f"   - Page {i+1}/{len(images)}: Running Tesseract...")
            text = extract_text_from_image(img)
            # Add a marker that this was OCR'd
            text = f"[OCR Extracted]\n{text}"
            page_texts.append(text)
            
        return page_texts
            
    except Exception as e:
        print(f"❌ Failed to convert PDF to images: {e}")
        print("   (Ensure Poppler is installed and in your PATH)")
        return []

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        print(ocr_pdf(fpath))
    else:
        print("Usage: python ocr_engine.py <path_to_pdf_or_image>")
