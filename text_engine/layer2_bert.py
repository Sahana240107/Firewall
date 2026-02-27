import sys
import os

# Add ml/ folder to path so predict.py can find its model
ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml')
sys.path.insert(0, os.path.abspath(ml_path))

from predict import predict as bert_predict

def analyze(text: str) -> dict:
    result = bert_predict(text)
    return {
        "label":      result["predicted_label"],
        "confidence": result["confidence"],
        "action":     result["action"],
        "redacted":   result["redacted_text"],
        "risk_score": result["risk_score"],
    }