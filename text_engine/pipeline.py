from text_engine import layer1_regex, layer2_bert, layer3_gemini


def run(text: str, source: str = "text") -> dict:
    """
    Master pipeline — ALL layers always run.
    Each layer redacts first, passes sanitized text to next layer.
    Gemini never sees original sensitive data.

    source = "text" | "image" | "docs" | "audio" | "video"
    """

    result = {
        "source":        source,
        "original_text": text,
        "redacted_text": text,
        "action":        "ALLOW",
        "risk_score":    0.0,
        "triggered_by":  [],
        "detections":    [],
        "layers_used":   [],
        "privacy_note":  ""
    }

    current_text = text

    # ── LAYER 1: Regex ─────────────────────────────────────
    result["layers_used"].append("regex")
    regex_matches = layer1_regex.detect(current_text)

    if regex_matches:
        current_text = layer1_regex.redact(current_text, regex_matches)
        result["action"] = "REDACT"
        result["risk_score"] = max(result["risk_score"], 0.97)
        result["triggered_by"].append("layer1_regex")
        result["detections"] += [
            {"type": k, "values": v, "source": "regex"}
            for k, v in regex_matches.items()
        ]

    result["redacted_text"] = current_text

    # ── LAYER 2: DistilBERT ────────────────────────────────
    result["layers_used"].append("distilbert")
    bert = layer2_bert.analyze(current_text)

    if bert["action"] in ("REDACT", "BLOCK"):
        if bert["redacted"]:
            current_text = bert["redacted"]
        result["action"] = bert["action"] if bert["action"] == "BLOCK" else "REDACT"
        result["risk_score"] = max(result["risk_score"], bert["risk_score"])
        result["triggered_by"].append(f"layer2_bert:{bert['label']}")
        result["detections"].append({
            "type":       bert["label"],
            "confidence": bert["confidence"],
            "source":     "distilbert"
        })

    result["redacted_text"] = current_text

    # ── LAYER 3: Gemini ────────────────────────────────────
    result["layers_used"].append("gemini")

    try:
        gemini = layer3_gemini.analyze(current_text)

        if gemini.get("detections"):
            gemini_redacted = gemini.get("redacted_text", "").strip()

            # FIX: Removed the `!= current_text` guard — it was silently
            # dropping Gemini's redaction whenever text was whitespace-equivalent
            # or Gemini returned a slightly reformatted version of the same text.
            # Now we always accept Gemini's redacted_text if it exists and is non-empty.
            if gemini_redacted:
                current_text = gemini_redacted

            result["action"] = "REDACT"
            result["risk_score"] = max(result["risk_score"], gemini.get("risk_score", 0) / 100)
            result["triggered_by"].append("layer3_gemini")
            result["detections"] += [
                {**d, "source": "gemini"}
                for d in gemini.get("detections", [])
            ]

        result["redacted_text"] = current_text
        result["privacy_note"] = (
            "Layers 1+2 redacted locally first. "
            "Gemini only received pre-sanitized text — original PII never transmitted."
        )

    except Exception as e:
        result["privacy_note"] = (
            f"Gemini unavailable ({str(e)[:60]}). "
            "Layers 1+2 redaction still applied."
        )

    if not result["triggered_by"]:
        result["action"] = "ALLOW"
        result["redacted_text"] = None

    return result
