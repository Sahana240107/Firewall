# layer3_gemini.py
# NOTE: Despite the filename (kept for import compatibility),
# this now uses Groq's LLaMA 3.3 70B as the Layer 3 AI backend.
# Reason: Gemini free tier quota is very limited (exhausts quickly).
# Groq free tier: 14,400 requests/day, 6000 tokens/min — much more generous.
# You already have GROQ_API_KEY in your .env from the audio pipeline.

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a privacy redaction AI. Your only job is to find PII in text and return JSON.

RETURN ONLY THIS JSON FORMAT — no markdown, no explanation, nothing else:
{
  "detections": [
    {
      "type": "phone",
      "value": "+91 98765 43210",
      "reason": "Indian mobile with country code",
      "sensitivity": "HIGH"
    }
  ],
  "redacted_text": "complete input text with every PII replaced by [TYPE] placeholders",
  "risk_score": 85,
  "summary": "Found 3 PII items"
}

STRICT RULES:
1. "value" must be the EXACT string from the input — copy it character for character
2. "redacted_text" must be the COMPLETE input text, with every PII replaced
3. If nothing is found, return empty detections array and original text unchanged

WHAT TO DETECT:
- Names of real people (even inside sentences like "my friend Ravi Kumar")
- Phone numbers in any format: +91 98765 43210, 9876543210, +1-800-555-0199
- Email addresses
- Physical addresses: street number, road name, city, pincode/zipcode
- Aadhaar numbers (12 digits), PAN cards (ABCDE1234F format)
- Bank account numbers, IFSC codes, UPI IDs
- Dates of birth
- Passwords or secrets mentioned in text
- Any other personal data

CONTEXTUAL RULES (important for accuracy):
- "+91 98765 43210" in any context = CRITICAL phone
- "14, 3rd Cross Street, Velachery, Chennai – 600042" = CRITICAL address
- A person's name mentioned alongside their PII = HIGH
- Generic business name without personal data = LOW (don't flag)
- Already-redacted placeholders like [EMAIL] = ignore, already handled

sensitivity: CRITICAL | HIGH | MEDIUM | LOW"""


def analyze(text: str) -> dict:
    # Skip if text is already fully redacted or too short
    if not text or not text.strip() or text.strip() == "[SENSITIVE CONTENT REDACTED]":
        return {
            "detections": [],
            "redacted_text": text,
            "risk_score": 0,
            "summary": "Skipped — text already redacted or empty"
        }

    try:
        client = get_client()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Analyze this text for PII:\n\n{text}"}
            ],
            temperature=0.0,   # deterministic — no creativity needed
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or ""
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except Exception:
                    return {
                        "detections": [],
                        "redacted_text": text,
                        "risk_score": 0,
                        "summary": f"Groq parse error: {raw[:150]}"
                    }
            else:
                return {
                    "detections": [],
                    "redacted_text": text,
                    "risk_score": 0,
                    "summary": f"Groq returned no JSON: {raw[:150]}"
                }

        # Ensure required fields exist
        if "detections" not in parsed:
            parsed["detections"] = []
        if "redacted_text" not in parsed or not parsed["redacted_text"]:
            parsed["redacted_text"] = text

        return parsed

    except Exception as e:
        return {
            "detections": [],
            "redacted_text": text,
            "risk_score": 0,
            "summary": f"Groq error: {str(e)}"
        }