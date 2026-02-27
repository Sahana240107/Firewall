from text_engine import layer1_regex, layer2_bert, layer3_gemini


def run(text: str, source: str = "text") -> dict:
    """
    Master pipeline — takes any text regardless of source.
    source = "text" | "image" | "docs" | "audio" | "video"
    """

    result = {
        "source": source,
        "original_text": text,
        "redacted_text": None,
        "action": "ALLOW",
        "risk_score": 0.0,
        "triggered_by": [],
        "detections": [],
        "layers_used": [],
        "privacy_note": ""
    }

    # ── LAYER 1: Regex ─────────────────────────────────────
    regex_matches = layer1_regex.detect(text)
    result["layers_used"].append("regex")

    if regex_matches:
        result["redacted_text"] = layer1_regex.redact(text, regex_matches)
        result["action"] = "REDACT"
        result["risk_score"] = 0.97
        result["triggered_by"].append("layer1_regex")
        result["detections"] = [
            {"type": k, "values": v, "source": "regex"}
            for k, v in regex_matches.items()
        ]
        result["privacy_note"] = "100% local — no data transmitted"
        return result  # stop here, no need for layer 2 or 3

    # ── LAYER 2: DistilBERT ────────────────────────────────
    bert = layer2_bert.analyze(text)
    result["layers_used"].append("distilbert")

    if bert["action"] in ("REDACT", "BLOCK"):
        result["redacted_text"] = bert["redacted"]
        result["action"] = bert["action"]
        result["risk_score"] = bert["risk_score"]
        result["triggered_by"].append(f"layer2_bert:{bert['label']}")
        result["detections"].append({
            "type": bert["label"],
            "confidence": bert["confidence"],
            "source": "distilbert"
        })
        result["privacy_note"] = "100% local — no data transmitted"
        return result  # stop here, no need for layer 3

    # ── LAYER 3: Gemini (only if 1+2 both pass) ────────────
    result["layers_used"].append("gemini")

    try:
        gemini = layer3_gemini.analyze(text)
        result["redacted_text"] = gemini.get("redacted_text", text)
        result["risk_score"] = gemini.get("risk_score", 0) / 100
        result["detections"] = gemini.get("detections", [])

        if gemini.get("detections"):
            result["action"] = "REDACT"
            result["triggered_by"].append("layer3_gemini")
        else:
            result["action"] = "ALLOW"
            result["redacted_text"] = None

        result["privacy_note"] = "Layers 1+2 passed. Gemini used as fallback."

    except Exception as e:
        result["action"] = "ALLOW"
        result["redacted_text"] = None
        result["triggered_by"].append(f"gemini_failed:{str(e)}")
        result["privacy_note"] = "Gemini unavailable, layers 1+2 passed clean"

    return result