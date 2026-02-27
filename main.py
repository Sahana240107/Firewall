from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from text_engine.pipeline import run
import uvicorn

app = FastAPI(title="PrivacyShield")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── TEXT TAB (you handle directly) ────────────────────────
class TextInput(BaseModel):
    text: str

@app.post("/api/scan/text")
def scan_text(body: TextInput):
    return run(body.text, source="text")


# ── IMAGE TAB (friend A extracts text, sends here) ────────
class ImageTextInput(BaseModel):
    extracted_text: str     # friend A sends this after OCR
    file_name: str = ""

@app.post("/api/scan/image")
def scan_image(body: ImageTextInput):
    return run(body.extracted_text, source="image")


# ── DOCS TAB (friend B extracts text, sends here) ─────────
class DocsTextInput(BaseModel):
    extracted_text: str     # friend B sends this after PDF parse
    file_name: str = ""

@app.post("/api/scan/docs")
def scan_docs(body: DocsTextInput):
    return run(body.extracted_text, source="docs")


# ── AUDIO TAB (friend C extracts text, sends here) ────────
class AudioTextInput(BaseModel):
    extracted_text: str     # friend C sends this after Whisper
    file_name: str = ""

@app.post("/api/scan/audio")
def scan_audio(body: AudioTextInput):
    return run(body.extracted_text, source="audio")


# ── VIDEO TAB (friend D extracts text, sends here) ────────
class VideoTextInput(BaseModel):
    extracted_text: str     # friend D sends this after frame OCR
    file_name: str = ""

@app.post("/api/scan/video")
def scan_video(body: VideoTextInput):
    return run(body.extracted_text, source="video")


# ── HEALTH CHECK ──────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline": ["regex", "distilbert", "gemini"],
        "endpoints": ["/api/scan/text", "/api/scan/image",
                      "/api/scan/docs", "/api/scan/audio",
                      "/api/scan/video"]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)