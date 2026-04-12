# 🛡️ PrivacyShield — AI Privacy Firewall

> A three-layer AI pipeline that automatically detects and redacts Personally Identifiable Information (PII) from text, documents, images, audio, and video — before you share anything.

---

## ✨ Features

- **Multi-modal scanning** — text, PDFs, DOCX, spreadsheets, images (via OCR), audio (via Whisper), and video frames
- **Three-layer detection pipeline** — Regex → DistilBERT → Gemini API (escalating fallback)
- **India-first PII support** — Aadhaar, PAN, Indian phone numbers, alongside universal patterns (email, credit card, SSN, DOB)
- **Format-preserving export** — redacted output returned in the same format as input (`.docx`, `.csv`, `.xlsx`, `.txt`)
- **Audio bleep engine** — timestamped PII in speech is replaced with beep tones; user controls which segments to mute
- **Privacy by default** — Layers 1 and 2 run 100% locally; Gemini is only called as a last resort
- **FastAPI backend** — clean REST endpoints, interactive Swagger docs, CORS enabled for frontend integration

---

## 🏗️ Architecture

```
Input (any format)
      │
      ▼
┌─────────────────────────────────────┐
│  Layer 1 · Regex                    │  ← credit cards, Aadhaar, PAN, phone,
│  (instant, zero network)            │    email, SSN, DOB, card numbers
└─────────────────┬───────────────────┘
                  │ if uncertain
                  ▼
┌─────────────────────────────────────┐
│  Layer 2 · DistilBERT               │  ← contextual PII, runs 100% locally
│  (sahana-24/ai-firewall-model)      │    labels: SAFE · PII · SENSITIVE
└─────────────────┬───────────────────┘
                  │ if still uncertain
                  ▼
┌─────────────────────────────────────┐
│  Layer 3 · Gemini API               │  ← fallback only; fired when
│  (cloud, optional)                  │    layers 1 + 2 both pass
└─────────────────┬───────────────────┘
                  │
                  ▼
         Redaction Engine
                  │
                  ▼
        Human Review → Export
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- `ffmpeg` (required for audio processing)
- `tesseract-ocr` (required for image OCR)
- A free [Gemini API key](https://aistudio.google.com) (optional — only used as fallback)

### 1. Clone the repository

```bash
git clone https://github.com/Sahana240107/Firewall.git
cd Firewall
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here   # optional — only used by Layer 3
GROQ_API_KEY=your_groq_key_here       # required for audio transcription
```

> **Note:** Never commit `.env` to version control. It is already listed in `.gitignore`.

### 5. Start the server

```bash
python main.py
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | API base |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/health` | Server health check |

> **First run:** The DistilBERT model (`sahana-24/ai-firewall-model`) will auto-download from HuggingFace (~1 minute). It is cached locally after that.

---

## 📁 Project Structure

```
Firewall/
├── adapters/
│   ├── audio_adapter.py       ← Groq Whisper transcription + LLaMA PII detection
│   ├── audio_processor.py     ← beep/bleep engine (apply_bleeps, convert_to_wav)
│   ├── docs_adapter.py        ← PDF, DOCX, CSV, XLSX text extraction
│   └── image_adapter.py       ← OCR text extraction wrapper
│
├── ml/
│   ├── model/                 ← auto-downloaded from HuggingFace on first run
│   ├── predict.py             ← DistilBERT inference (SAFE / PII / SENSITIVE)
│   └── train.py               ← fine-tuning script (already completed)
│
├── text_engine/
│   ├── __init__.py
│   ├── layer1_regex.py        ← pattern library (Aadhaar, PAN, email, card, etc.)
│   ├── layer2_bert.py         ← DistilBERT wrapper
│   ├── layer3_gemini.py       ← Gemini API fallback
│   └── pipeline.py            ← orchestrates all three layers
│
├── main.py                    ← FastAPI application (all endpoints)
├── redactron.html             ← standalone browser UI
├── test_pipeline.py           ← integration tests
├── requirements.txt
└── .env                       ← secrets (DO NOT commit)
```

---

## 🔌 API Reference

### Scan Endpoints

| Endpoint | Method | Input | Description |
|----------|--------|-------|-------------|
| `/api/scan/text` | POST | `{ "text": "..." }` | Scan raw text |
| `/api/scan/image` | POST | `{ "extracted_text": "...", "file_name": "..." }` | Scan OCR output from an image |
| `/api/scan/docs` | POST | `{ "extracted_text": "...", "file_name": "..." }` | Scan pre-extracted document text |
| `/api/scan/docs-upload` | POST | `multipart/form-data` file | Upload and scan a document directly |
| `/api/scan/video` | POST | `{ "extracted_text": "...", "file_name": "..." }` | Scan OCR text from video frames |
| `/api/scan/audio-file` | POST | `multipart/form-data` file | Upload audio → auto-bleep all PII → return redacted WAV |

### Audio (Two-Step Flow)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audio/process` | POST | Upload audio → transcribe → detect PII → return timestamped segments |
| `/api/audio/export` | POST | Upload audio + segment list → apply selected bleeps → return redacted audio |

### Export

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/export/redacted-doc` | POST | Upload original file + redacted text → return format-preserving redacted file |

### Utilities

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Pipeline status and available endpoints |
| `/api/debug-audio` | POST | Return raw transcript + timestamps without applying bleeps |

---

## 📨 Response Format

Every scan endpoint returns a consistent JSON payload:

```json
{
  "action": "REDACT",
  "redacted_text": "My card is [CREDIT CARD] and email is [EMAIL]",
  "risk_score": 0.97,
  "triggered_by": ["layer1_regex"],
  "layers_used": ["regex"],
  "detections": [
    {
      "type": "credit_card",
      "values": ["4532 1234 1234 5678"],
      "source": "regex",
      "severity": "HIGH"
    }
  ],
  "privacy_note": "100% local — no data transmitted"
}
```

| Field | Description |
|-------|-------------|
| `action` | `ALLOW`, `REDACT`, or `BLOCK` |
| `redacted_text` | Input text with PII replaced by `[TYPE]` tokens |
| `risk_score` | Confidence score from 0.0 to 1.0 |
| `triggered_by` | Which layer(s) detected PII |
| `detections` | Array of detected PII items with type, value, source, and severity |
| `privacy_note` | Indicates whether any data left the device |

---

## 🔗 Integration Example

Any service can connect by POSTing its extracted text to the appropriate endpoint:

```python
import requests

result = requests.post("http://localhost:8000/api/scan/image", json={
    "extracted_text": "Patient John Doe, Aadhaar: 1234 5678 9012, DOB: 12Jan1990",
    "file_name": "patient_card.jpg"
})

data = result.json()

if data["action"] == "REDACT":
    print("Redacted output:", data["redacted_text"])
    print("Detected:", data["detections"])
```

---

## 🧪 Running Tests

```bash
# Terminal 1 — start the server
python main.py

# Terminal 2 — run the test suite
python test_pipeline.py
```

---

## 🤖 ML Model

| Property | Value |
|----------|-------|
| **HuggingFace repo** | `sahana-24/ai-firewall-model` |
| **Base architecture** | DistilBERT (fine-tuned for sequence classification) |
| **Labels** | `SAFE (0)` · `PII (1)` · `SENSITIVE (2)` |
| **Setup** | Auto-downloads on first run — no manual steps needed |

---

## 🧩 Supported File Types

| Category | Formats |
|----------|---------|
| Documents | `.pdf`, `.docx`, `.txt`, `.md`, `.json` |
| Spreadsheets | `.csv`, `.xls`, `.xlsx` |
| Audio | `.wav`, `.mp3`, and any format `ffmpeg` supports |
| Images | Any format supported by Tesseract OCR |

---

## 📦 Dependencies

```
fastapi          uvicorn          pydantic
transformers     torch            huggingface_hub
groq             soundfile        numpy
pdfplumber       python-docx      pandas
opencv-python    pytesseract      python-dotenv
requests         imageio-ffmpeg
```

---

## 🔐 Security & Privacy Notes

- **Layer 1 (Regex) and Layer 2 (DistilBERT) run entirely on your machine** — no data leaves the device.
- **Layer 3 (Gemini)** is only invoked when the first two layers are both uncertain. You can disable it entirely by omitting the `GEMINI_API_KEY`.
- The `.env` file is excluded from version control via `.gitignore`. Never commit API keys.
- Audio processing uses Groq's Whisper API for transcription — audio files are sent to Groq's servers for that step.

---

## 🗺️ Roadmap

- [ ] Docker / containerised deployment
- [ ] Configurable redaction token labels (e.g. `████` instead of `[EMAIL]`)
- [ ] Webhook support for async large-file processing
- [ ] Dashboard UI with redaction history and audit log
- [ ] Support for real-time video stream scanning

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change or add.

---

## 📄 License

This project is open source. See `LICENSE` for details.
