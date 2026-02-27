"""
redact_pipeline.py — OCR + PII Redaction + Face Blur Pipeline
══════════════════════════════════════════════════════════════

Integrates with your FastAPI backend (Layer1 regex + Layer2 DistilBERT +
Layer3 Gemini) at localhost:8000.  Optionally runs fully offline with
client-side regex fallback.

NEW in v3.0:
  • Face detection via OpenCV Haar cascades → Gaussian blur redaction
  • PII text → solid black bar redaction
  • Redaction manifest JSON saved alongside output so regions can be
    "de-redacted" by the review tool (redact_review.py)
  • Two-pass export:  (a) fully-redacted image  (b) review JSON

Flow:
  1. OCR the image with Tesseract
  2. Send extracted text to FastAPI /api/scan/image
  3. Match returned PII tokens → OCR bounding boxes → black bars
  4. Run OpenCV face detection → Gaussian blur on each face
  5. Save redacted image + manifest

Install:
    pip install pytesseract opencv-python requests numpy

    Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    Haar cascade (ships with opencv-python — no manual download needed)

Usage:
    # Full pipeline (backend + face blur):
    python redact_pipeline.py <image_path> [options]

    # Skip backend, supply PII tokens manually:
    python redact_pipeline.py <image_path> --tokens "John Smith" "412-83-9021"

    # Disable face blur:
    python redact_pipeline.py <image_path> --no-face

    # Disable PII text redaction:
    python redact_pipeline.py <image_path> --no-pii

    # Debug: show all detected boxes (green=OCR, red=PII, blue=face):
    python redact_pipeline.py <image_path> --debug

    # De-redact using manifest (review mode):
    python redact_pipeline.py --review <manifest_json>

Examples:
    python redact_pipeline.py id_card.jpg
    python redact_pipeline.py id_card.jpg --debug --out result.jpg
    python redact_pipeline.py id_card.jpg --tokens "Angela Greene" "5843"
    python redact_pipeline.py --review id_card_manifest.json
"""

import cv2
import numpy as np
import pytesseract
import json
import sys
import argparse
import requests
import copy
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Path to Tesseract (Windows — adjust if needed; on Mac/Linux it's usually auto-found)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BACKEND_URL     = "http://localhost:8000"   # Your FastAPI server
BACKEND_TIMEOUT = 30                        # seconds

# Extra pixels padding around every redaction box
REDACTION_PADDING = 4

# Minimum Tesseract confidence (0–100)
MIN_CONFIDENCE = 20

# OCR language(s)
OCR_LANGUAGE = "eng"

# Face blur settings
FACE_BLUR_RADIUS   = 31    # Must be odd. Higher = stronger blur.
FACE_HAAR_SCALE    = 1.1   # detectMultiScale scaleFactor
FACE_HAAR_NEIGHBORS = 4    # detectMultiScale minNeighbors (lower = more detections)

# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# REGEX FALLBACK — mirrors layer1_regex.py (used when backend is offline)
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

REGEX_PATTERNS = {
    "credit_card":  (_re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  "HIGH"),
    "aadhaar":      (_re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                    "HIGH"),
    "pan":          (_re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),                    "HIGH"),
    "ssn":          (_re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        "HIGH"),
    "email":        (_re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "MEDIUM"),
    "phone":        (_re.compile(r"\b(\(\d{3}\)\s?)?\d{3}[\s.\-]?\d{4}\b|\b\d{10}\b"), "MEDIUM"),
}

def regex_fallback(text: str) -> list[dict]:
    """Return list of {type, value, severity, source} dicts."""
    results = []
    for ptype, (pattern, severity) in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            results.append({
                "type":     ptype,
                "value":    m.group(0),
                "severity": severity,
                "source":   "regex_client"
            })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — OCR
# ══════════════════════════════════════════════════════════════════════════════

def ocr_image(image_path: str) -> dict:
    """Run Tesseract; return structured result with per-word bboxes."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        lang=OCR_LANGUAGE,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )

    text_blocks = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not word or conf < MIN_CONFIDENCE:
            continue
        x, y, bw, bh = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        text_blocks.append({
            "id":   len(text_blocks),
            "text": word,
            "confidence": round(conf / 100, 4),
            "bbox": {"x": x, "y": y, "w": bw, "h": bh},
        })

    text_blocks.sort(key=lambda b: (b["bbox"]["y"], b["bbox"]["x"]))

    return {
        "image_path":   str(Path(image_path).resolve()),
        "image_size":   {"w": w, "h": h},
        "full_text":    " ".join(b["text"] for b in text_blocks),
        "total_blocks": len(text_blocks),
        "text_blocks":  text_blocks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BACKEND / FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def get_pii_tokens(full_text: str, manual_tokens: list[str] | None) -> list[dict]:
    """
    Returns list of {token, type, layer} dicts to redact.
    Priority: manual_tokens → backend API → regex fallback.
    """
    if manual_tokens is not None:
        return [{"token": t, "type": "manual", "layer": "manual"} for t in manual_tokens if t.strip()]

    # Try backend
    try:
        print(f"\n--- TEXT BEING SENT TO BACKEND ---")
        print(full_text)
        print(f"----------------------------------\n")
        r = requests.post(
            BACKEND_URL + "/api/scan/image",
            json={"extracted_text": full_text, "file_name": "pipeline_input"},
            timeout=BACKEND_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        print(f"  Backend response — action: {data.get('action')} | triggered: {data.get('triggered_by')}")

        tokens = []
        for det in data.get("detections", []):
            vals = det.get("values") or ([det["value"]] if det.get("value") else [])
            for v in vals:
                tokens.append({"token": v, "type": det.get("type","unknown"), "layer": det.get("source","pipeline")})

        # If backend returned tagged text, extract tokens from it
        if not tokens and data.get("redacted_text") and data.get("action") == "REDACT":
            tokens = extract_tokens_from_tagged(full_text, data["redacted_text"])

        return tokens

    except requests.exceptions.ConnectionError:
        print(f"  ⚠  Backend offline ({BACKEND_URL}) — using local regex fallback")
    except Exception as e:
        print(f"  ⚠  Backend error: {e} — using local regex fallback")

    # Regex fallback
    dets = regex_fallback(full_text)
    return [{"token": d["value"], "type": d["type"], "layer": "regex_fallback"} for d in dets]


def extract_tokens_from_tagged(original: str, tagged: str) -> list[dict]:
    """Recover original token values from [TAG] substituted text."""
    TAG_RE = _re.compile(r"\[([^\[\]]+)\]")
    tag_matches = list(TAG_RE.finditer(tagged))
    if not tag_matches:
        return []
    tokens, o_cursor = [], 0
    for idx, match in enumerate(tag_matches):
        tag_start, tag_end, tag_label = match.start(), match.end(), match.group(1)
        prev_end  = tag_matches[idx-1].end() if idx > 0 else 0
        literal   = tagged[prev_end:tag_start]
        if literal:
            pos = original.find(literal, o_cursor)
            if pos != -1: o_cursor = pos + len(literal)
        suffix = tagged[tag_end:tag_matches[idx+1].start()] if idx+1 < len(tag_matches) else tagged[tag_end:]
        if suffix:
            ep = original.find(suffix, o_cursor)
            if ep != -1: span, o_cursor = original[o_cursor:ep].strip(), ep
            else:        span, o_cursor = original[o_cursor:].strip(), len(original)
        else:
            span, o_cursor = original[o_cursor:].strip(), len(original)
        for word in span.split():
            tokens.append({"token": word, "type": tag_label, "layer": "pipeline"})
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TOKEN MATCHING → PII BBOXES
# ══════════════════════════════════════════════════════════════════════════════

def match_tokens_to_blocks(text_blocks: list, tokens: list[dict]) -> list[dict]:
    """Match redacted tokens to OCR word bboxes. Returns redaction region list."""
    if not tokens:
        return []
    tokens_meta = [
        {"clean": t["token"].lower().replace(r"[^a-z0-9@._\-]",""), **t}
        for t in tokens
    ]
    regions = []
    for block in text_blocks:
        btext = block["text"].strip()
        bclean = btext.lower().rstrip(".,;:!?\"'()")
        matched_meta = next(
            (t for t in tokens_meta
             if t["clean"].replace(r"[^a-z0-9@._\-]","") in bclean
             or bclean in t["clean"].replace(r"[^a-z0-9@._\-]","")
             or bclean == t["clean"].replace(r"[^a-z0-9@._\-]","")),
            None
        )
        if matched_meta:
            bb = block["bbox"]
            regions.append({
                "type":  "pii",
                "label": matched_meta["type"].upper(),
                "text":  btext,
                "layer": matched_meta["layer"],
                "x":     bb["x"], "y": bb["y"],
                "w":     bb["w"], "h": bb["h"],
                "state": "redacted"   # for manifest / review
            })
    return regions


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FACE DETECTION (OpenCV Haar cascades → Gaussian blur)
# ══════════════════════════════════════════════════════════════════════════════

def detect_faces(image: np.ndarray) -> list[dict]:
    """
    Detect faces using OpenCV's frontal face Haar cascade.
    Returns list of {x, y, w, h, state} dicts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # improve detection in varied lighting

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        print("  ⚠  Haar cascade not found — skipping face detection")
        return []

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor  = FACE_HAAR_SCALE,
        minNeighbors = FACE_HAAR_NEIGHBORS,
        minSize      = (30, 30),
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )

    regions = []
    if len(faces) == 0:
        return regions

    h, w = image.shape[:2]
    for (fx, fy, fw, fh) in faces:
        # Cast numpy int32 → Python int so JSON serialization works
        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
        pad_x = int(fw * 0.15)
        pad_y = int(fh * 0.15)
        regions.append({
            "type":  "face",
            "label": "FACE",
            "text":  "face region",
            "layer": "opencv_haar",
            "x":     max(0, fx - pad_x),
            "y":     max(0, fy - pad_y),
            "w":     min(w - fx + pad_x, fw + pad_x * 2),
            "h":     min(h - fy + pad_y, fh + pad_y * 2),
            "state": "redacted"
        })

    return regions


def apply_gaussian_blur(image: np.ndarray, x: int, y: int, w: int, h: int, radius: int) -> np.ndarray:
    """Blur a rectangular region of an image in-place. Returns modified image."""
    if w < 2 or h < 2:
        return image
    # Ensure radius is odd and at least 3
    r = max(3, radius | 1)
    roi     = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (r, r), 0)
    # Apply multiple passes for stronger blur
    blurred = cv2.GaussianBlur(blurred, (r, r), 0)
    blurred = cv2.GaussianBlur(blurred, (r, r), 0)
    image[y:y+h, x:x+w] = blurred
    return image


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — APPLY REDACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_redactions(
    image: np.ndarray,
    regions: list[dict],
    padding: int = REDACTION_PADDING,
) -> np.ndarray:
    """
    Apply redactions to a copy of the image.
    PII regions  → solid black rectangle
    Face regions → Gaussian blur
    Returns redacted image (original is NOT modified).
    """
    out = image.copy()
    h, w = out.shape[:2]

    for r in regions:
        if r.get("state") == "revealed":
            continue   # skip revealed regions (for review mode)

        x1 = max(0, r["x"] - (padding if r["type"]=="pii" else 0))
        y1 = max(0, r["y"] - (padding if r["type"]=="pii" else 0))
        x2 = min(w, r["x"] + r["w"] + (padding if r["type"]=="pii" else 0))
        y2 = min(h, r["y"] + r["h"] + (padding if r["type"]=="pii" else 0))
        rw, rh = x2-x1, y2-y1

        if r["type"] == "face":
            out = apply_gaussian_blur(out, x1, y1, rw, rh, FACE_BLUR_RADIUS)
            print(f"  [FACE BLURRED]  bbox ({x1},{y1}) → ({x2},{y2})")
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)
            print(f"  [PII REDACTED]  '{r['text']}'  [{r['label']}]  bbox ({x1},{y1}) → ({x2},{y2})")

    return out


def save_debug_image(image: np.ndarray, regions: list[dict], output_path: str) -> str:
    """
    Annotated debug image:
      Green  = PII redaction boxes
      Blue   = face blur boxes
    """
    dbg = image.copy()
    for r in regions:
        x1,y1,x2,y2 = r["x"],r["y"],r["x"]+r["w"],r["y"]+r["h"]
        color = (255, 0, 0) if r["type"] == "face" else (0, 200, 0)
        cv2.rectangle(dbg, (x1,y1), (x2,y2), color, 2)
        label = f"[{r['label']}] {r['text'][:20]}"
        cv2.putText(dbg, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    p = Path(output_path)
    debug_path = str(p.parent / (p.stem + "_debug" + p.suffix))
    cv2.imwrite(debug_path, dbg)
    return debug_path


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW MODE — de-redact specific regions using manifest
# ══════════════════════════════════════════════════════════════════════════════

def review_mode(manifest_path: str) -> None:
    """
    Interactive CLI review: load manifest, toggle regions, re-export.
    This mirrors the browser's click-to-reveal but works from the terminal.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    image_path = manifest["image_path"]
    regions    = manifest["regions"]
    orig       = cv2.imread(image_path)
    if orig is None:
        print(f"ERROR: Could not load original image: {image_path}")
        sys.exit(1)

    print(f"\nReview Mode — {len(regions)} region(s)")
    print("=" * 60)
    for i, r in enumerate(regions):
        print(f"  [{i}] type={r['type']}  label={r['label']}  text='{r['text']}'  state={r['state']}")

    print("\nEnter region numbers to toggle (comma-separated), or 'all', or 'done':")
    while True:
        inp = input("  > ").strip().lower()
        if inp == "done":
            break
        if inp == "all":
            for r in regions:
                r["state"] = "revealed" if r["state"] == "redacted" else "redacted"
        else:
            try:
                idxs = [int(x.strip()) for x in inp.split(",") if x.strip().isdigit()]
                for idx in idxs:
                    if 0 <= idx < len(regions):
                        r = regions[idx]
                        r["state"] = "revealed" if r["state"] == "redacted" else "redacted"
                        print(f"    [{idx}] {r['label']} → {r['state']}")
            except ValueError:
                print("    Invalid input. Try: 0,1,3  or  all  or  done")
        # Re-export
        out = apply_redactions(orig, regions)
        p   = Path(image_path)
        out_path = str(p.parent / (p.stem + "_reviewed" + p.suffix))
        cv2.imwrite(out_path, out)
        print(f"    Saved: {out_path}")

    print("Review complete.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path:    str,
    manual_tokens: list[str] | None = None,
    output_path:   str | None       = None,
    enable_pii:    bool             = True,
    enable_face:   bool             = True,
    save_debug:    bool             = False,
    save_manifest: bool             = True,
) -> dict:
    """
    Full pipeline. Returns summary dict.
    Also saves a manifest JSON so regions can be reviewed/de-redacted later.
    """
    print(f"\n{'═'*60}")
    print(f"  Redactron Pipeline v3.0")
    print(f"  Image  : {image_path}")
    print(f"  Modes  : PII={'on' if enable_pii else 'off'}  Face={'on' if enable_face else 'off'}")
    print(f"{'═'*60}\n")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    all_regions: list[dict] = []

    # ── STEP 1: OCR ──────────────────────────────────────────────────────────
    if enable_pii:
        print("[STEP 1] Running Tesseract OCR…")
        ocr = ocr_image(image_path)
        print(f"         {ocr['total_blocks']} word(s) extracted")
        print(f"         Full text: {ocr['full_text'][:200]}{'…' if len(ocr['full_text'])>200 else ''}\n")

        # ── STEP 2: PII tokens from backend / fallback ────────────────────────
        print("[STEP 2] Getting PII detections…")
        pii_tokens = get_pii_tokens(ocr["full_text"], manual_tokens)
        print(f"         {len(pii_tokens)} token(s) to redact: {[t['token'] for t in pii_tokens]}\n")

        # ── STEP 3: Match tokens → OCR bboxes ────────────────────────────────
        print("[STEP 3] Matching tokens to OCR bounding boxes…")
        pii_regions = match_tokens_to_blocks(ocr["text_blocks"], pii_tokens)
        print(f"         {len(pii_regions)} PII region(s) matched\n")
        all_regions.extend(pii_regions)
    else:
        print("[STEP 1-3] PII detection disabled — skipping\n")

    # ── STEP 4: Face detection ────────────────────────────────────────────────
    if enable_face:
        print("[STEP 4] Running OpenCV face detection…")
        face_regions = detect_faces(image)
        print(f"         {len(face_regions)} face(s) detected\n")
        all_regions.extend(face_regions)
    else:
        print("[STEP 4] Face detection disabled — skipping\n")

    # ── STEP 5: Apply redactions ──────────────────────────────────────────────
    print("[STEP 5] Applying redactions…")
    redacted = apply_redactions(image, all_regions)

    p = Path(image_path)
    out_path = output_path or str(p.parent / (p.stem + "_redacted" + p.suffix))
    cv2.imwrite(out_path, redacted)
    print(f"\n         Redacted image saved: {out_path}")

    if save_debug:
        dbg_path = save_debug_image(image, all_regions, out_path)
        print(f"         Debug image saved  : {dbg_path}")

    # ── STEP 6: Save manifest (for review/de-redaction) ───────────────────────
    manifest = {
        "image_path":    str(Path(image_path).resolve()),
        "output_path":   str(Path(out_path).resolve()),
        "total_regions": len(all_regions),
        "pii_count":     sum(1 for r in all_regions if r["type"]=="pii"),
        "face_count":    sum(1 for r in all_regions if r["type"]=="face"),
        "regions":       all_regions,
    }
    if save_manifest:
        manifest_path = str(p.parent / (p.stem + "_manifest.json"))
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)
        print(f"         Manifest saved      : {manifest_path}")
        print(f"         → Run review mode   : python redact_pipeline.py --review {manifest_path}")

    print(f"\n{'═'*60}")
    print(f"  Done.  {len(all_regions)} region(s) redacted  "
          f"({manifest['pii_count']} PII + {manifest['face_count']} face(s))")
    print(f"{'═'*60}\n")

    return {"status": "ok", "output_path": out_path, "manifest": manifest}


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR + PII + Face blur redaction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image_path", nargs="?",
        help="Path to input image (JPG, PNG, etc.)")
    parser.add_argument("--review", metavar="MANIFEST_JSON",
        help="Enter interactive review / de-redact mode using a saved manifest")
    parser.add_argument("--out", metavar="PATH",
        help="Output path for redacted image")
    parser.add_argument("--tokens", metavar="TOKEN", nargs="+",
        help="Supply PII tokens manually — skip backend API")
    parser.add_argument("--no-pii",  action="store_true",
        help="Disable PII text detection and redaction")
    parser.add_argument("--no-face", action="store_true",
        help="Disable face detection and blur")
    parser.add_argument("--debug",   action="store_true",
        help="Save annotated debug image (green=PII, blue=face)")
    parser.add_argument("--no-manifest", action="store_true",
        help="Do not save manifest JSON")
    parser.add_argument("--lang",    metavar="LANG", default=OCR_LANGUAGE,
        help=f'Tesseract language (default: "{OCR_LANGUAGE}")')
    parser.add_argument("--padding", metavar="PX", type=int, default=REDACTION_PADDING,
        help=f"Extra pixels around PII boxes (default: {REDACTION_PADDING})")
    parser.add_argument("--blur-radius", metavar="R", type=int, default=FACE_BLUR_RADIUS,
        help=f"Face Gaussian blur kernel radius (odd number, default: {FACE_BLUR_RADIUS})")
    parser.add_argument("--backend", metavar="URL", default=BACKEND_URL,
        help=f"FastAPI backend URL (default: {BACKEND_URL})")

    args = parser.parse_args()

    # Apply CLI overrides to config
    OCR_LANGUAGE     = args.lang
    REDACTION_PADDING = args.padding
    FACE_BLUR_RADIUS  = args.blur_radius | 1  # ensure odd
    BACKEND_URL       = args.backend

    # Review mode
    if args.review:
        review_mode(args.review)
        sys.exit(0)

    if not args.image_path:
        parser.error("image_path is required unless using --review")

    result = run_pipeline(
        image_path    = args.image_path,
        manual_tokens = args.tokens,
        output_path   = args.out,
        enable_pii    = not args.no_pii,
        enable_face   = not args.no_face,
        save_debug    = args.debug,
        save_manifest = not args.no_manifest,
    )

    sys.exit(0 if result["status"] == "ok" else 1)
