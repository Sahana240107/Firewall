import os
import json
import tempfile
import pandas as pd
import pdfplumber
from docx import Document


def extract_text_from_document(file_path: str) -> str:
    """
    Called by pipeline.py
    Pass any document file path → get plain text back
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _read_txt(file_path)
    elif ext == ".csv":
        return _read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return _read_excel(file_path)
    elif ext == ".pdf":
        return _read_pdf(file_path)
    elif ext == ".docx":
        return _read_docx(file_path)
    elif ext == ".json":
        return _read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Called by main.py (FastAPI upload)
    Pass raw bytes + filename → get plain text back
    """
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return extract_text_from_document(tmp_path)
    finally:
        os.unlink(tmp_path)


# =====================================================
# Internal readers (don't touch these)
# =====================================================
def _read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_csv(path):
    df = pd.read_csv(path)
    return df.to_string()

def _read_excel(path):
    df = pd.read_excel(path)
    return df.to_string()

def _read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def _read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)