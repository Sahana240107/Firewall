import re

PATTERNS = {
    "credit_card":      r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "aadhaar":          r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":              r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "email":            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:\.[A-Za-z]{2,})?\b",  # FIX: TLD now optional so krithika@oksbi matches
    "phone":            r"\b\d{10}\b",
    "ssn":              r"\b\d{3}-\d{2}-\d{4}\b",
    "ip_address":       r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "api_key":          r"\b[A-Za-z0-9]{32,}\b",
    "passport":         r"\b[A-Z]{1,2}\d{6,9}\b",
    # NEW patterns
    "account_number":   r"\b\d{9,18}\b",                                              # FIX: catches 548392016284
    "ifsc":             r"\b[A-Z]{4}0[A-Z0-9]{6}\b",                                  # FIX: catches SBIN0001234
    "dob":              r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",                    # FIX: catches 21/08/2004
    "upi":              r"\b[\w.\-]+@[a-zA-Z]+\b",                                    # FIX: catches krithika@oksbi
}

PLACEHOLDERS = {
    "credit_card":    "[CREDIT CARD]",
    "aadhaar":        "[AADHAAR]",
    "pan":            "[PAN]",
    "email":          "[EMAIL]",
    "phone":          "[PHONE]",
    "ssn":            "[SSN]",
    "ip_address":     "[IP ADDRESS]",
    "api_key":        "[API KEY]",
    "passport":       "[PASSPORT]",
    "account_number": "[ACCOUNT NUMBER]",
    "ifsc":           "[IFSC]",
    "dob":            "[DOB]",
    "upi":            "[UPI ID]",
}

# FIX: Expanded to also match "Name: Krithika Palani" style labels
NAME_PATTERN = re.compile(
    r"(?:my name is|i am|i'm|name\s*:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    re.IGNORECASE
)


def detect(text: str) -> dict:
    """Returns all regex matches found"""
    matches = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            matches[key] = found
    # Detect names separately
    name_matches = NAME_PATTERN.findall(text)
    if name_matches:
        matches["name"] = name_matches
    return matches


def redact(text: str, matches: dict) -> str:
    """Replaces matched PII with placeholders"""
    redacted = text
    for key, pattern in PATTERNS.items():
        if key in matches:
            redacted = re.sub(pattern, PLACEHOLDERS.get(key, "[REDACTED]"), redacted)

    # Redact names
    def replace_name(match):
        return match.group(0).replace(match.group(1), "[NAME]")
    redacted = NAME_PATTERN.sub(replace_name, redacted)

    return redacted
