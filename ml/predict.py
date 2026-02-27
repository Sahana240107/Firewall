import torch
import re
import numpy as np
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ==============================
# CONFIG
# ==============================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
HF_REPO   = "sahana-24/ai-firewall-model"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# AUTO DOWNLOAD FROM HUGGINGFACE
# if model folder is missing or empty
# ==============================
def ensure_model_exists():
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"[INFO] Model not found locally. Downloading from HuggingFace: {HF_REPO}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO, local_dir=MODEL_DIR)
        print(f"[INFO] Model downloaded to {MODEL_DIR}")
    else:
        print(f"[INFO] Model found at {MODEL_DIR}")

ensure_model_exists()

# ==============================
# LOAD MODEL + TOKENIZER
# ==============================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# ==============================
# LABEL MAP
# ==============================
label_map = {
    0: "SAFE",
    1: "PII",
    2: "SENSITIVE"
}

# ==============================
# REGEX RULES (Layer 1 backup)
# ==============================
regex_patterns = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "email":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone":   r"\b\d{10}\b",
    "api_key": r"\b[A-Za-z0-9]{32,}\b"
}

placeholder_map = {
    "aadhaar": "[AADHAAR]",
    "email":   "[EMAIL]",
    "phone":   "[PHONE]",
    "api_key": "[API_KEY]"
}

name_pattern = re.compile(
    r"(?:my name is|i am|i'm|name\s*:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
    re.IGNORECASE
)


def rule_based_detection(text):
    matches = {}
    for key, pattern in regex_patterns.items():
        found = re.findall(pattern, text)
        if found:
            matches[key] = found
    return matches


def redact_text(text, rule_matches):
    redacted = text
    for key, pattern in regex_patterns.items():
        placeholder = placeholder_map.get(key, "[REDACTED]")
        redacted = re.sub(pattern, placeholder, redacted)

    def replace_name(match):
        return match.group(0).replace(match.group(1), "[NAME]")
    redacted = name_pattern.sub(replace_name, redacted)
    return redacted


def calculate_risk(label, confidence, rule_matches):
    if rule_matches:
        return 0.95, "BLOCK"
    if label == "SENSITIVE" and confidence > 0.90:
        return confidence, "BLOCK"
    if label == "PII" and confidence > 0.80:
        return confidence, "REDACT"
    return confidence, "ALLOW"


def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits        = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probabilities)
    confidence      = float(probabilities[predicted_class])
    label           = label_map[predicted_class]

    rule_matches        = rule_based_detection(text)
    risk_score, action  = calculate_risk(label, confidence, rule_matches)

    redacted = None
    if action in ("BLOCK", "REDACT"):
        redacted = redact_text(text, rule_matches)

    return {
        "input_text":      text,
        "predicted_label": label,
        "confidence":      round(confidence, 4),
        "risk_score":      round(risk_score, 4),
        "action":          action,
        "rule_matches":    rule_matches,
        "redacted_text":   redacted
    }


# ==============================
# CLI TESTING
# ==============================
if __name__ == "__main__":
    print("\n🔥 AI Firewall Prediction Console")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == "exit":
            break

        result = predict(user_input)
        print("\n--- RESULT ---")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("----------------\n")