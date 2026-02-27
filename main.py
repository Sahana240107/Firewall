from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from text_engine.pipeline import run
import uvicorn
from fastapi.responses import FileResponse
from adapters.audio_adapter import process_audio
from audio_processor import apply_bleeps
import os

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

#--------------audio-----------
@app.post("/api/scan/audio-file")
async def scan_audio_file(file: UploadFile = File(...)):
    tmp_dir = "temp"
    os.makedirs(tmp_dir, exist_ok=True)

    original_ext = os.path.splitext(file.filename)[1].lower()
    inp_path = os.path.join(tmp_dir, f"input_{file.filename}")
    wav_path = os.path.join(tmp_dir, "input_converted.wav")
    out_path = os.path.join(tmp_dir, "redacted_output.wav")

    try:
        # Save uploaded file
        with open(inp_path, "wb") as f:
            f.write(await file.read())

        # Always convert to wav using imageio-ffmpeg
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[Convert] Converting {original_ext} → wav using imageio-ffmpeg...")
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", inp_path, wav_path],
            check=True, capture_output=True
        )
        print("[Convert] Done!")

        # Run full pipeline on wav file
        result = process_audio(wav_path)

        if result["action"] in ("REDACT", "BLOCK") and result["segments"]:
            apply_bleeps(wav_path, out_path, result["segments"])
            return FileResponse(
                out_path,
                media_type="audio/wav",
                filename=f"redacted_{file.filename}.wav"
            )
        else:
            return {
                "action": result["action"],
                "message": "No sensitive content detected",
                "transcript": result["transcript"],
                "layers_used": result["layers_used"]
            }

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        for path in [inp_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)


#-----------temporary-------------

@app.post("/api/debug-audio")
async def debug_audio(file: UploadFile = File(...)):
    """Debug endpoint to see raw timestamps."""
    tmp_dir = "temp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    original_ext = os.path.splitext(file.filename)[1].lower()
    inp_path = os.path.join(tmp_dir, f"input_{file.filename}")
    wav_path = os.path.join(tmp_dir, "debug_converted.wav")

    try:
        with open(inp_path, "wb") as f:
            f.write(await file.read())

        import imageio_ffmpeg, subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, "-y", "-i", inp_path, wav_path], check=True, capture_output=True)

        result = process_audio(wav_path)
        return result

    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        for path in [inp_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)