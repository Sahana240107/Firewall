import re

PATTERNS = {
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "aadhaar":     r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone":       r"\b\d{10}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "ip_address":  r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "api_key":     r"\b[A-Za-z0-9]{32,}\b",
    "passport":    r"\b[A-Z]{1,2}\d{6,9}\b",
}

PLACEHOLDERS = {
    "credit_card": "[CREDIT CARD]",
    "aadhaar":     "[AADHAAR]",
    "pan":         "[PAN]",
    "email":       "[EMAIL]",
    "phone":       "[PHONE]",
    "ssn":         "[SSN]",
    "ip_address":  "[IP ADDRESS]",
    "api_key":     "[API KEY]",
    "passport":    "[PASSPORT]",
}

NAME_PATTERN = re.compile(
    r"(?:my name is|i am|i'm|name\s*:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
    re.IGNORECASE
)


def detect(text: str) -> dict:
    """Returns all regex matches found"""
    matches = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, text)
        if found:
            matches[key] = found
    return matches


def redact(text: str, matches: dict) -> str:
    """Replaces matched PII with placeholders"""
    redacted = text
    for key, pattern in PATTERNS.items():
        if key in matches:
            redacted = re.sub(pattern, PLACEHOLDERS.get(key, "[REDACTED]"), redacted)
    
    def replace_name(match):
        return match.group(0).replace(match.group(1), "[NAME]")
    redacted = NAME_PATTERN.sub(replace_name, redacted)
    
    return redacted