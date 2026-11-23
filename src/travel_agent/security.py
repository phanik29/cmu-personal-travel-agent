"""Safety helpers: untrusted-text sanitization, tool-role checks, secret redaction."""
from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from typing import Any

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)", re.I),
    re.compile(r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)", re.I),
    re.compile(r"(?:reveal|print|show)\s+(?:the\s+|your\s+)?(?:system|developer)\s+prompt", re.I),
    re.compile(r"\byou\s+(?:must|should|will)\s+now\b[^.]*", re.I),
    re.compile(r"\b(?:book|purchase|pay\s+for|cancel)\s+(?:the|this|a|my|all)\b[^.]*\bimmediately\b", re.I),
    re.compile(r"<\s*/?\s*(?:script|iframe|object|embed|style)[^>]*>", re.I),
    re.compile(r"\b(?:system|assistant)\s*:\s*", re.I),
)
_SECRET_KEY_HINTS = ("api_key", "apikey", "token", "secret", "password", "authorization", "auth")

# Name tokens that identify tools able to change real-world state.
ACTION_NAME_TOKENS = frozenset({
    "book", "booking_create", "pay", "payment", "purchase", "cancel", "cancellation", "delete", "remove",
    "send", "modify", "update", "create", "prebook", "refund", "charge", "checkout", "reserve", "confirm",
    "draft", "label", "write", "post",
})

READ_ROLES = frozenset({"search", "qa"})
EXECUTOR_ROLE = "executor"


def sanitize_untrusted_text(text: Any, max_length: int = 500) -> str:
    """Treat external text as data: strip control characters and neutralize instruction-like phrases."""
    if text is None:
        return ""
    cleaned = _CONTROL_CHARS.sub(" ", str(text))
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[removed]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 1].rstrip() + "…"
    return cleaned


def escape_for_display(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def tool_name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return {token for token in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if token}


def is_read_only_tool_name(name: str) -> bool:
    return not (tool_name_tokens(name) & ACTION_NAME_TOKENS)


def assert_tool_allowed_for_role(tool_name: str, role: str) -> None:
    """Read roles may never call a tool whose name implies an action, even if misconfigured."""
    if role == EXECUTOR_ROLE:
        return
    if role not in READ_ROLES:
        raise PermissionError(f"Unknown tool role {role!r}")
    if not is_read_only_tool_name(tool_name):
        raise PermissionError(f"Tool {tool_name!r} can change state and is not available to the {role!r} role")


def _is_secret_key(key: str, extra_keys: Iterable[str]) -> bool:
    lowered = key.lower()
    return key in extra_keys or any(hint in lowered for hint in _SECRET_KEY_HINTS)


def redact_secrets(data: Mapping[str, Any], extra_keys: Iterable[str] = ()) -> dict[str, Any]:
    extra = frozenset(extra_keys)
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _is_secret_key(key, extra):
            redacted[key] = "***" if value else None
        elif isinstance(value, Mapping):
            redacted[key] = redact_secrets(value, extra)
        else:
            redacted[key] = value
    return redacted


def mask_secrets_in_text(text: str, secrets: Iterable[str | None]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "***")
    return text


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_ALLOWED_TEST_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
_NOT_PERSONAL = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?|\b[A-Za-z]+-[0-9A-Fa-f]{6,}\b")


def find_personal_data(text: str) -> list[str]:
    """Return categories of personal data found in text (sandbox example.com addresses are allowed)."""
    found: list[str] = []
    emails = [match for match in _EMAIL.findall(text) if not match.lower().endswith(_ALLOWED_TEST_EMAIL_DOMAINS)]
    if emails:
        found.append("email")
    text = _NOT_PERSONAL.sub(" ", text)  # dates, times, and our ids (trip-…, run-…, SBX-…) are not personal data
    if _CARD.search(text):
        found.append("card_number")
    elif _PHONE.search(text):
        found.append("phone")
    return found
