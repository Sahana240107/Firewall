import os
import json
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def transcribe(audio_path: str) -> tuple:
    """Transcribe audio, return (text, word_timestamps)"""
    print(f"[Groq Whisper] Transcribing {audio_path}...")
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en"
        )
    return transcription.text, transcription.words


def map_to_timestamps(detections: list, word_timestamps: list) -> list:
    """Map detected PII text back to audio timestamps."""
    result = []
    words = [w["word"].lower().strip(".,!?-") for w in word_timestamps]

    for det in detections:
        # Handle both pipeline formats
        # Layer 1 returns: {"type": "phone", "values": ["9876543210"]}
        # Layer 2/3 returns: {"type": "PII", "value": "..."}
        values = det.get("values") or [det.get("value", "")]
        det_type = det.get("type", "UNKNOWN")

        for val in values:
            if not val:
                continue

            sensitive_words = str(val).lower().split()
            sensitive_words = [w.strip(".,!?-") for w in sensitive_words]

            best_start = None
            best_end = None

            # Strategy 1: exact full match
            for i in range(len(words) - len(sensitive_words) + 1):
                if words[i:i + len(sensitive_words)] == sensitive_words:
                    best_start = word_timestamps[i]["start"]
                    best_end = word_timestamps[i + len(sensitive_words) - 1]["end"]
                    break

            # Strategy 2: first + last word
            if best_start is None and len(sensitive_words) > 1:
                for i, w in enumerate(words):
                    if w == sensitive_words[0]:
                        search_range = min(i + len(sensitive_words) + 5, len(words))
                        for j in range(i + 1, search_range):
                            if words[j] == sensitive_words[-1]:
                                best_start = word_timestamps[i]["start"]
                                best_end = word_timestamps[j]["end"]
                                break
                    if best_start is not None:
                        break

            # Strategy 3: first word + estimate
            if best_start is None:
                for i, w in enumerate(words):
                    if w == sensitive_words[0]:
                        best_start = word_timestamps[i]["start"]
                        end_idx = min(i + len(sensitive_words) - 1, len(words) - 1)
                        best_end = word_timestamps[end_idx]["end"]
                        break

            if best_start is not None:
                result.append({
                    "text": val,
                    "start": best_start,
                    "end": best_end,
                    "type": det_type,
                })
                print(f"[Timestamp] {det_type}: '{val}' → {best_start}s–{best_end}s")
            else:
                print(f"[Timestamp] Could not map '{val}' — skipping")

    return result


def process_audio(audio_path: str) -> dict:
    from text_engine.pipeline import run
    import re
    import json

    llama_client = client 

    LLAMA_PROMPT = """
You are a privacy-protection AI. Analyze this transcript for sensitive/private information.

CRITICAL CONTEXT RULE:
- 16-digit number on a CREDIT/DEBIT CARD = sensitive
- Tracking number, order ID, serial number = NOT sensitive
- Always use surrounding words as context

Detect: CREDIT_CARD, PHONE, AADHAAR, PAN, SSN, ADDRESS, EMAIL, PASSWORD, NAME

Return ONLY valid JSON:
{
  "segments": [
    {
      "text": "exact sensitive words",
      "type": "CREDIT_CARD",
      "reason": "why sensitive",
      "confidence": 0.95
    }
  ]
}
If nothing sensitive, return {"segments": []}
"""

    # Step 1: Transcribe
    transcript_text, word_timestamps = transcribe(audio_path)
    print(f"[Transcript] {transcript_text}")

    # Step 2: LLaMA context-aware detection first
    print("[LLaMA] Running context-aware detection...")
    response = llama_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": LLAMA_PROMPT},
            {"role": "user", "content": f"Transcript: {transcript_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    raw = response.choices[0].message.content
    try:
        llama_result = json.loads(raw)
    except Exception:
        llama_result = {"segments": []}

    llama_segments = llama_result.get("segments", [])
    print(f"[LLaMA] Found {len(llama_segments)} sensitive segments")

    # Step 3: Convert LLaMA segments to pipeline detection format
    detections = [
        {
            "type": s["type"],
            "values": [s["text"]],
            "source": "llama"
        }
        for s in llama_segments
    ]

    # Step 4: Run regex only for types LLaMA didn't catch
    # But skip credit card regex — trust LLaMA for context
    from text_engine import layer1_regex
    regex_matches = layer1_regex.detect(transcript_text)

    llama_types = [s["type"].lower() for s in llama_segments]
    for key, values in regex_matches.items():
        # Skip credit card from regex — LLaMA handles context
        if key == "credit_card":
            continue
        # Skip if LLaMA already found this type
        if key in llama_types:
            continue
        detections.append({
            "type": key.upper(),
            "values": values,
            "source": "regex"
        })
        print(f"[Regex] Extra: {key} = {values}")

    # Step 5: Map to timestamps
    segments = map_to_timestamps(detections, word_timestamps)

    # Determine action
    action = "REDACT" if segments else "ALLOW"

    return {
        "transcript": transcript_text,
        "redacted_text": None,
        "action": action,
        "risk_score": 0.97 if segments else 0.0,
        "layers_used": ["llama", "regex"],
        "segments": segments
    }
