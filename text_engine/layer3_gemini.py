from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

PROMPT = """You are a privacy analysis AI for text content.

Analyze the text for PII and sensitive data.
Return ONLY valid JSON, no markdown:

{
  "detections": [
    {
      "type": "credit_card",
      "value": "masked value",
      "reason": "why this is sensitive",
      "sensitivity": "CRITICAL"
    }
  ],
  "redacted_text": "full text with [REDACTED] placeholders",
  "risk_score": 87,
  "summary": "Found 2 sensitive items"
}

sensitivity: CRITICAL | HIGH | MEDIUM | LOW

CONTEXTUAL INTELLIGENCE:
- 16-digit on credit card = CRITICAL, on shipping label = LOW
- Name on business card = LOW, in medical record = HIGH
- OTP codes = CRITICAL, Passwords = CRITICAL
- Home address = CRITICAL, business address = LOW

Detect: names, Aadhaar, PAN, credit cards, CVV, phones,
emails, addresses, passwords, OTPs, account numbers,
medical info, passport numbers."""


def analyze(text: str) -> dict:
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{PROMPT}\n\nText to analyze:\n{text}"
        )
        raw = response.text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {
                "detections": [],
                "redacted_text": text,
                "risk_score": 0,
                "summary": "Gemini parse error"
            }
    except Exception as e:
        return {
            "detections": [],
            "redacted_text": text,
            "risk_score": 0,
            "summary": f"Gemini error: {str(e)}"
        }
