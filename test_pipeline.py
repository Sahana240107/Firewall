"""
PrivacyShield - Text Pipeline Test Script
=========================================
Run this from your project root (where main.py is):
    python test_pipeline.py

It tests:
1. Layer 1 (regex) in isolation
2. Layer 2 (DistilBERT) in isolation  
3. Full pipeline via API (needs server running)
4. Edge cases
"""

import sys
import os
import json
import re

print("=" * 60)
print("  PrivacyShield - Pipeline Test")
print("=" * 60)

# ─────────────────────────────────────────────
# TEST 1: Layer 1 Regex (no model needed)
# ─────────────────────────────────────────────
print("\n[TEST 1] Layer 1 - Regex Detection")
print("-" * 40)

# Copy of your regex patterns to test standalone
PATTERNS = {
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "aadhaar":     r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone":       r"\b\d{10}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
}

test_cases_regex = [
    {
        "name": "Credit Card",
        "text": "My card number is 4532 1234 1234 5678 please process it",
        "expect": "credit_card"
    },
    {
        "name": "Email",
        "text": "Contact me at john.doe@gmail.com for details",
        "expect": "email"
    },
    {
        "name": "Phone Number",
        "text": "Call me at 9876543210 anytime",
        "expect": "phone"
    },
    {
        "name": "Aadhaar",
        "text": "My Aadhaar number is 1234 5678 9012",
        "expect": "aadhaar"
    },
    {
        "name": "PAN Card",
        "text": "PAN: ABCDE1234F is my tax ID",
        "expect": "pan"
    },
    {
        "name": "SSN",
        "text": "Social security: 123-45-6789",
        "expect": "ssn"
    },
    {
        "name": "Safe text (should detect nothing)",
        "text": "The weather today is sunny and warm",
        "expect": None
    },
    {
        "name": "Multiple PII in one text",
        "text": "Email: test@example.com and phone 9876543210",
        "expect": "email"  # at minimum
    },
]

passed = 0
failed = 0

for tc in test_cases_regex:
    matches = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, tc["text"])
        if found:
            matches[key] = found

    if tc["expect"] is None:
        ok = len(matches) == 0
    else:
        ok = tc["expect"] in matches

    status = "✅ PASS" if ok else "❌ FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    print(f"  {status} | {tc['name']}")
    if not ok:
        print(f"         Expected: {tc['expect']}")
        print(f"         Got:      {list(matches.keys())}")
    if matches:
        print(f"         Detected: {list(matches.keys())}")

print(f"\nLayer 1 Results: {passed} passed, {failed} failed")


# ─────────────────────────────────────────────
# TEST 2: Layer 2 - DistilBERT
# ─────────────────────────────────────────────
print("\n[TEST 2] Layer 2 - DistilBERT Model")
print("-" * 40)

# Try to import from ml/ folder
try:
    sys.path.insert(0, os.path.join(os.getcwd(), 'ml'))
    from predict import predict
    print("  ✅ Model loaded successfully\n")

    bert_tests = [
        {
            "name": "Clear PII - credit card context",
            "text": "Please charge my Visa card 4532123412345678",
            "expect_not": "ALLOW"
        },
        {
            "name": "Sensitive - medical info",
            "text": "Patient has HIV diagnosis and is on medication",
            "expect_not": "ALLOW"
        },
        {
            "name": "Sensitive - password",
            "text": "My password is SuperSecret123!",
            "expect_not": "ALLOW"
        },
        {
            "name": "Safe - normal sentence",
            "text": "I love to eat pizza on weekends",
            "expect": "ALLOW"
        },
        {
            "name": "Safe - public business info",
            "text": "The meeting is at 10am in conference room B",
            "expect": "ALLOW"
        },
    ]

    b_passed = 0
    b_failed = 0

    for tc in bert_tests:
        result = predict(tc["text"])
        action = result["action"]
        label = result["predicted_label"]
        conf = result["confidence"]

        if "expect_not" in tc:
            ok = action != tc["expect_not"]
        else:
            ok = action == tc["expect"]

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            b_passed += 1
        else:
            b_failed += 1

        print(f"  {status} | {tc['name']}")
        print(f"         Action: {action} | Label: {label} | Confidence: {conf}")
        if result.get("redacted_text"):
            print(f"         Redacted: {result['redacted_text'][:80]}...")

    print(f"\nLayer 2 Results: {b_passed} passed, {b_failed} failed")

except Exception as e:
    print(f"  ⚠️  Could not load model: {e}")
    print("  Make sure you're running from project root")
    print("  and ml/model/ folder exists")


# ─────────────────────────────────────────────
# TEST 3: Full Pipeline (if pipeline.py exists)
# ─────────────────────────────────────────────
print("\n[TEST 3] Full Pipeline - All Layers")
print("-" * 40)

try:
    sys.path.insert(0, os.getcwd())
    from text_engine.pipeline import run

    pipeline_tests = [
        {
            "name": "Credit card → should stop at Layer 1",
            "text": "My card is 4532 1234 1234 5678",
            "expect_layer": "layer1_regex",
            "expect_action": "REDACT"
        },
        {
            "name": "Email → should stop at Layer 1",
            "text": "Send invoice to billing@company.com",
            "expect_layer": "layer1_regex",
            "expect_action": "REDACT"
        },
        {
            "name": "Phone → should stop at Layer 1",
            "text": "Call 9876543210 for support",
            "expect_layer": "layer1_regex",
            "expect_action": "REDACT"
        },
        {
            "name": "Contextual PII → Layer 2 catches it",
            "text": "I am sharing my personal diagnosis with you",
            "expect_action": "REDACT"
        },
        {
            "name": "Safe text → ALLOW",
            "text": "Today is a beautiful sunny day",
            "expect_action": "ALLOW"
        },
        {
            "name": "Multiple PII types",
            "text": "Name: Rahul Sharma, phone: 9876543210, email: rahul@test.com",
            "expect_action": "REDACT"
        },
    ]

    p_passed = 0
    p_failed = 0

    for tc in pipeline_tests:
        result = run(tc["text"])
        action = result["action"]
        triggered = result.get("triggered_by", [])
        layers = result.get("layers_used", [])
        redacted = result.get("redacted_text")

        action_ok = action == tc["expect_action"]
        layer_ok = True
        if "expect_layer" in tc:
            layer_ok = tc["expect_layer"] in triggered

        ok = action_ok and layer_ok
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            p_passed += 1
        else:
            p_failed += 1

        print(f"  {status} | {tc['name']}")
        print(f"         Action:     {action}")
        print(f"         Triggered:  {triggered}")
        print(f"         Layers:     {layers}")
        if redacted:
            print(f"         Redacted:   {redacted[:100]}")

    print(f"\nPipeline Results: {p_passed} passed, {p_failed} failed")

except ImportError as e:
    print(f"  ⚠️  Pipeline not found: {e}")
    print("  Make sure text_engine/pipeline.py exists")


# ─────────────────────────────────────────────
# TEST 4: API Endpoint Test
# ─────────────────────────────────────────────
print("\n[TEST 4] API Endpoint Test")
print("-" * 40)
print("  Make sure your server is running:")
print("  > python main.py")
print()

try:
    import urllib.request
    import urllib.error

    url = "http://localhost:8000/health"
    req = urllib.request.urlopen(url, timeout=3)
    data = json.loads(req.read())
    print(f"  ✅ Server is running: {data}")

    # Test the text endpoint
    api_tests = [
        {
            "name": "Text scan - credit card",
            "payload": {"text": "My card 4532 1234 1234 5678 expires soon"},
            "expect_action": "REDACT"
        },
        {
            "name": "Text scan - safe",
            "payload": {"text": "Good morning everyone"},
            "expect_action": "ALLOW"
        },
    ]

    for tc in api_tests:
        try:
            data_bytes = json.dumps(tc["payload"]).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:8000/api/scan/text",
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            response = urllib.request.urlopen(req, timeout=10)
            result = json.loads(response.read())
            action = result.get("action", "unknown")
            ok = action == tc["expect_action"]
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status} | {tc['name']} → {action}")
            if result.get("redacted_text"):
                print(f"         Redacted: {result['redacted_text']}")
        except Exception as e:
            print(f"  ❌ FAIL | {tc['name']} → {e}")

except Exception as e:
    print(f"  ⚠️  Server not running ({e})")
    print("  Start it with: python main.py")
    print("  Then re-run this test")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
print("""
Quick fix guide:
  ❌ Layer 1 fails   → check text_engine/layer1_regex.py patterns
  ❌ Layer 2 fails   → check ml/model/ folder exists, sys.path correct
  ❌ Pipeline fails  → check text_engine/pipeline.py imports
  ❌ API fails       → run: python main.py  then re-test

Your folder should look like:
  privacy-firewall/
  ├── ml/
  │   ├── model/        ← trained model files
  │   └── predict.py
  ├── text_engine/
  │   ├── __init__.py   ← IMPORTANT: needs to exist (can be empty)
  │   ├── layer1_regex.py
  │   ├── layer2_bert.py
  │   ├── layer3_gemini.py
  │   └── pipeline.py
  ├── main.py
  └── test_pipeline.py  ← this file
""")