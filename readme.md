# 🛡️ PrivacyShield — AI Privacy Firewall

Automatically detects and redacts PII from text, images, audio, and documents before sharing.

---

## 🏗️ Architecture

```
Input (any format)
      ↓
Layer 1: Regex          → catches credit cards, Aadhaar, PAN, phone, email instantly
      ↓ (if missed)
Layer 2: DistilBERT     → catches contextual PII, runs 100% locally
      ↓ (if uncertain)
Layer 3: Gemini API     → fallback only, fired when layers 1+2 both pass
      ↓
Redaction Engine
      ↓
Human Review → Export
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/privacy-firewall
cd privacy-firewall
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
Create a `.env` file in the root:
```
GEMINI_API_KEY=your_key_here
```
Get free key from: https://aistudio.google.com

### 5. Run the server
```bash
python main.py
```

Server runs at: http://localhost:8000
API docs at:    http://localhost:8000/docs

> **Note:** The ML model will auto-download from HuggingFace on first run.
> This takes ~1 minute. After that it's cached locally.

---

## 📁 Folder Structure

```
privacy-firewall/
├── ml/
│   ├── model/              ← auto downloaded from HuggingFace
│   ├── predict.py          ← DistilBERT inference
│   └── train.py            ← model training (already done)
├── text_engine/
│   ├── __init__.py
│   ├── layer1_regex.py     ← regex PII detection
│   ├── layer2_bert.py      ← DistilBERT wrapper
│   ├── layer3_gemini.py    ← Gemini API fallback
│   └── pipeline.py         ← combines all 3 layers
├── adapters/
│   ├── image_adapter.py    ← for image friend's OCR output
│   ├── docs_adapter.py     ← for docs friend's output
│   └── audio_adapter.py    ← for audio friend's output
├── main.py                 ← FastAPI server
├── test_pipeline.py        ← run tests
├── requirements.txt
└── .env                    ← NOT pushed to github (add your key here)
```

---

## 🔌 API Endpoints

| Endpoint | Method | Who Uses It |
|---|---|---|
| `/api/scan/text` | POST | Direct text input |
| `/api/scan/image` | POST | Image friend (after OCR) |
| `/api/scan/docs` | POST | Docs friend (after PDF parse) |
| `/api/scan/audio` | POST | Audio friend (after Whisper) |
| `/api/scan/video` | POST | Video friend (after frame OCR) |
| `/health` | GET | Check server status |

---

## 📨 How Friends Connect Their Work

Every friend sends their extracted text to their endpoint:

```python
import requests

result = requests.post("http://localhost:8000/api/scan/image", json={
    "extracted_text": "your OCR extracted text here",
    "file_name": "my_image.jpg"
})

print(result.json())
```

### Response format:
```json
{
  "action": "REDACT",
  "redacted_text": "My card is [CREDIT CARD] and email is [EMAIL]",
  "risk_score": 0.97,
  "triggered_by": ["layer1_regex"],
  "layers_used": ["regex"],
  "detections": [
    {"type": "credit_card", "values": ["4532 1234 1234 5678"], "source": "regex"}
  ],
  "privacy_note": "100% local — no data transmitted"
}
```

---

## 🧪 Run Tests

```bash
# Terminal 1 — start server
python main.py

# Terminal 2 — run tests
python test_pipeline.py
```

---

## 🤖 ML Model

Model hosted on HuggingFace: `sahana-24/ai-firewall-model`
- Architecture: DistilBERT fine-tuned for sequence classification
- Labels: SAFE (0), PII (1), SENSITIVE (2)
- Auto-downloads on first run — no manual setup needed

---

## 📦 Requirements

```
fastapi
uvicorn
transformers
torch
google-genai
python-dotenv
huggingface_hub
pydantic
```