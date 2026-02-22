"""
ocr_engine.py — Phase 2: OCR & Multi-Modal Text Extraction
────────────────────────────────────────────────────────────
Extracts text from PDFs, images, and charts using Sarvam AI Document Intelligence API.
Dependencies:
- wrappers: pip install requests
"""

import os
import requests
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

def check_dependencies() -> bool:
    """Checks if Sarvam API Key is available."""
    if not SARVAM_API_KEY:
        print("⚠️ SARVAM_API_KEY not found in .env. OCR will be disabled.")
        return False
    return True

def ocr_pdf(file_path: str) -> List[str]:
    """
    Submits a PDF or Image to Sarvam AI Document Intelligence API.
    Returns a list of strings (one per page where applicable, or a single combined string).
    """
    if not check_dependencies():
        return []

    print(f"📷 [OCR] Processing '{os.path.basename(file_path)}' with Sarvam AI (this may take a while)...")
    
    # Sarvam AI Document Intelligence endpoint
    url = "https://api.sarvam.ai/document-intelligence/job"
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }
    
    try:
        # Determine content type based on extension
        ext = os.path.splitext(file_path)[1].lower()
        content_type = "application/pdf" if ext == ".pdf" else "image/jpeg" # simplified
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, content_type)
            }
            # Start job
            response = requests.post(url, headers=headers, files=files)
            
        if response.status_code != 200:
            print(f"❌ Sarvam AI API Error: {response.text}")
            return []
            
        # Mocking or handling job completion polling if the API is async
        # (Assuming synchronous response for this example, typical for some simple OCR APIs, 
        # or assuming the response directly contains extracted text if it's a smaller payload).
        # *Note*: Real implementation may require polling a job status endpoint if Sarvam's API is fully async.
        
        # Taking a simplified approach assuming a JSON response with text
        data = response.json()
        
        # Example formatting - adjust based on actual Sarvam API schema
        extracted_text = data.get("text", "") 
        if not extracted_text:
            # Maybe it's page-based
            pages = data.get("pages", [])
            extracted_text = "\n\n".join([page.get("text", "") for page in pages])
            
        if not extracted_text:
           print(f"⚠️ No text extracted by Sarvam AI for {file_path}")
           return []
            
        # Add a marker that this was OCR'd
        text = f"[OCR Extracted via Sarvam AI]\n{extracted_text}"
        
        # For compatibility with ingest.py which expects a list of pages
        return [text]
            
    except Exception as e:
        print(f"❌ Failed to process file with Sarvam AI API: {e}")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        print(ocr_pdf(fpath))
    else:
        print("Usage: python ocr_engine.py <path_to_pdf_or_image>")
