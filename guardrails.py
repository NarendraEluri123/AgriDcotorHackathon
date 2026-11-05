"""
AgriDoctor – Guardrails Module
================================
Implements all 9 categories of guardrails for the agentic AI workflow:

  1.  Input Guardrails          – validate/sanitise user inputs
  2.  Output Guardrails         – validate AI-generated responses
  3.  Tool Execution Guardrails – control tool/API interactions
  4.  Data Guardrails           – protect data accessed by agents
  5.  Security Guardrails       – protect against adversarial attacks
  6.  Operational Guardrails    – control resource consumption
  7.  Human-in-the-Loop         – require human approval for critical actions
  8.  Multi-Agent Guardrails    – safe coordination between agents
  9.  Ethical & Compliance      – responsible AI behaviour + audit trail

Each guardrail returns a GuardrailResult(passed: bool, reason: str, sanitised_value).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging (audit trail)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GUARDRAIL] %(levelname)s – %(message)s",
)
_audit_log: List[Dict[str, Any]] = []          # in-memory audit trail


def _audit(category: str, check: str, passed: bool, detail: str = "") -> None:
    """Append one record to the in-memory audit trail."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "category": category,
        "check": check,
        "passed": passed,
        "detail": detail,
    }
    _audit_log.append(record)
    level = logging.INFO if passed else logging.WARNING
    logging.log(level, "[%s] %s → %s  %s", category, check, "PASS" if passed else "FAIL", detail)


def get_audit_log() -> List[Dict[str, Any]]:
    """Return a copy of the current audit trail."""
    return list(_audit_log)


def clear_audit_log() -> None:
    """Clear the in-memory audit trail (call at start of each session)."""
    _audit_log.clear()


# ---------------------------------------------------------------------------
# GuardrailResult – unified return type
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    reason: str
    sanitised_value: Any = None          # cleaned/safe version of the input

    def __bool__(self) -> bool:
        return self.passed


# ===========================================================================
# 1. INPUT GUARDRAILS
# ===========================================================================

# --- Prompt-injection patterns --------------------------------------------
_PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"forget\s+(everything|all|your\s+instructions?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!a\s+farmer|an?\s+agricultural)", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+(?!a\s+farmer|an?\s+agricultural)", re.I),
    re.compile(r"(disregard|override|bypass)\s+(your\s+)?(instructions?|guidelines?|rules?|constraints?)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\b", re.I),                       # "Do Anything Now" jailbreak
    re.compile(r"system\s*prompt\s*:", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[\s*INST\s*\]", re.I),              # Llama-style injection
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
]

# --- Malicious / dangerous content patterns --------------------------------
_MALICIOUS_PATTERNS: List[re.Pattern] = [
    re.compile(r"<script[\s>]", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"on\w+\s*=\s*[\"']", re.I),           # HTML event handlers
    re.compile(r"(\bexec\b|\beval\b|\bos\.system\b|\bsubprocess\b)", re.I),
    re.compile(r"(__import__|importlib)", re.I),
    re.compile(r"(rm\s+-rf|del\s+/[sq]|format\s+c:)", re.I),
]

# --- SQL injection patterns ------------------------------------------------
_SQL_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(--|;|\/\*|\*\/)", re.I),
    re.compile(r"\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b", re.I),
    re.compile(r"'\s*(OR|AND)\s*'?\d+'\s*=\s*'?\d+", re.I),
]

# --- PII detection patterns ------------------------------------------------
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email":        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone_intl":   re.compile(r"\+?\d[\d\s\-().]{7,}\d"),
    "credit_card":  re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "aadhaar":      re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "pan_india":    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ssn_us":       re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address":   re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# --- Inappropriate content -------------------------------------------------
_INAPPROPRIATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(fuck|shit|asshole|bitch|bastard|damn|crap|piss)\b", re.I),
    re.compile(r"\b(kill|murder|bomb|explode|weapon|terrorist)\b", re.I),
    re.compile(r"\b(porn|sex|nude|naked|xxx)\b", re.I),
]

# Maximum symptom text length (characters)
_MAX_INPUT_LENGTH = 2000


def validate_input(symptom_text: str) -> GuardrailResult:
    """
    Run all input guardrails on the symptom description text.
    Returns a GuardrailResult; if passed=False, do not forward to agents.
    """
    text = symptom_text or ""

    # ── 1a. Input length validation ─────────────────────────────────────────
    if len(text) > _MAX_INPUT_LENGTH:
        _audit("INPUT", "length_check", False, f"len={len(text)}")
        return GuardrailResult(False, f"Input is too long ({len(text)} chars). Maximum allowed is {_MAX_INPUT_LENGTH} characters.")
    _audit("INPUT", "length_check", True)

    # ── 1b. Normalise unicode (prevent lookalike attacks) ───────────────────
    text = unicodedata.normalize("NFKC", text)

    # ── 1c. Prompt injection detection ──────────────────────────────────────
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            _audit("INPUT", "prompt_injection", False, pattern.pattern[:60])
            return GuardrailResult(False, "Potential prompt-injection attempt detected. Please describe your crop symptoms only.")

    _audit("INPUT", "prompt_injection", True)

    # ── 1d. Malicious content detection ─────────────────────────────────────
    for pattern in _MALICIOUS_PATTERNS:
        if pattern.search(text):
            _audit("INPUT", "malicious_content", False, pattern.pattern[:60])
            return GuardrailResult(False, "Input contains potentially malicious content and has been blocked.")

    _audit("INPUT", "malicious_content", True)

    # ── 1e. PII detection ───────────────────────────────────────────────────
    pii_found = [label for label, pat in _PII_PATTERNS.items() if pat.search(text)]
    if pii_found:
        _audit("INPUT", "pii_detection", False, f"types={pii_found}")
        return GuardrailResult(False, f"Personal information ({', '.join(pii_found)}) detected in the description. Please remove any personal details before submitting.")

    _audit("INPUT", "pii_detection", True)

    # ── 1f. Inappropriate content filtering ─────────────────────────────────
    for pattern in _INAPPROPRIATE_PATTERNS:
        if pattern.search(text):
            _audit("INPUT", "inappropriate_content", False, pattern.pattern[:60])
            return GuardrailResult(False, "Input contains inappropriate language. Please keep the description professional and crop-related.")

    _audit("INPUT", "inappropriate_content", True)

    # ── 1g. Agricultural relevance check ────────────────────────────────────
    if text.strip():
        ag_keywords = [
            "leaf", "leave", "plant", "crop", "farm", "soil", "stem", "root",
            "flower", "fruit", "seed", "disease", "spot", "yellow", "brown",
            "wilt", "mould", "mold", "fungus", "pest", "insect", "blight",
            "rust", "rot", "lesion", "discolour", "discolor", "dry", "wet",
            "water", "drought", "growth", "wheat", "rice", "maize", "cotton",
            "tomato", "potato", "vegetable", "grass", "paddy", "soybean",
        ]
        text_lower = text.lower()
        has_ag_context = any(kw in text_lower for kw in ag_keywords)
        if not has_ag_context and len(text.split()) > 10:
            _audit("INPUT", "relevance_check", False, "no agricultural keywords")
            return GuardrailResult(False, "The description does not appear to be related to crop or plant symptoms. Please describe only what you observe on your crops or plants.")
    _audit("INPUT", "relevance_check", True)

    # Sanitise: strip HTML entities
    sanitised = html.escape(text)
    return GuardrailResult(True, "All input checks passed.", sanitised)


def validate_api_key_format(api_key: str) -> GuardrailResult:
    """Basic format check on the Replicate API key."""
    if not api_key or not api_key.strip():
        _audit("INPUT", "api_key_format", False, "empty")
        return GuardrailResult(False, "API key cannot be empty.")
    key = api_key.strip()
    # Replicate API keys are typically ~40 chars starting with "r8_"
    if len(key) < 20:
        _audit("INPUT", "api_key_format", False, f"len={len(key)}")
        return GuardrailResult(False, "API key appears too short. Please verify your Replicate API key.")
    if not re.match(r'^[\w\-]+$', key):
        _audit("INPUT", "api_key_format", False, "invalid chars")
        return GuardrailResult(False, "API key contains unexpected characters.")
    _audit("INPUT", "api_key_format", True)
    return GuardrailResult(True, "API key format valid.", key)


def validate_image_file(uploaded_file) -> GuardrailResult:
    """Validate the uploaded image file before processing."""
    if uploaded_file is None:
        return GuardrailResult(False, "No file uploaded.")

    # Check filename extension
    name = uploaded_file.name.lower()
    allowed_extensions = (".jpg", ".jpeg", ".png")
    if not any(name.endswith(ext) for ext in allowed_extensions):
        _audit("INPUT", "image_format", False, f"name={name}")
        return GuardrailResult(False, "Only JPG, JPEG, and PNG image files are accepted.")

    # Check file size (max 10 MB)
    max_size_bytes = 10 * 1024 * 1024
    uploaded_file.seek(0, 2)          # seek to end
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)             # reset
    if file_size > max_size_bytes:
        _audit("INPUT", "image_size", False, f"size={file_size}")
        return GuardrailResult(False, f"Image file is too large ({file_size // 1024 // 1024} MB). Maximum allowed is 10 MB.")

    # Check magic bytes for real image format
    header = uploaded_file.read(12)
    uploaded_file.seek(0)
    is_jpeg = header[:3] == b'\xff\xd8\xff'
    is_png  = header[:8] == b'\x89PNG\r\n\x1a\n'
    if not (is_jpeg or is_png):
        _audit("INPUT", "image_magic_bytes", False, f"header={header[:4].hex()}")
        return GuardrailResult(False, "File does not appear to be a valid image. Please upload a real JPG or PNG photo.")

    _audit("INPUT", "image_validation", True, f"size={file_size}")
    return GuardrailResult(True, "Image file valid.")


# ===========================================================================
# 2. OUTPUT GUARDRAILS
# ===========================================================================

# Minimum expected section headings in each agent's output
_PATHOLOGIST_REQUIRED_HEADINGS = [
    "Primary Diagnosis",
    "Observed Symptoms",
    "Confidence Level",
    "Disclaimer",
]

_EXTENSION_REQUIRED_HEADINGS = [
    "Summary for the Farmer",
    "Immediate Remedial Actions",
    "Safety Warnings",
]

# Hallucination indicators – phrases an LLM uses when fabricating
_HALLUCINATION_SIGNALS: List[re.Pattern] = [
    re.compile(r"as\s+(an?\s+)?ai\s+(language\s+)?model,?\s+i\s+(cannot|don'?t)", re.I),
    re.compile(r"i\s+don'?t\s+have\s+(real[\-\s]?time|access\s+to)", re.I),
    re.compile(r"i\s+apologize,?\s+but\s+i\s+(cannot|am\s+unable)", re.I),
    re.compile(r"note:\s+this\s+is\s+(completely\s+)?fictional", re.I),
    re.compile(r"i\s+made\s+up\s+the\s+(following|above)", re.I),
]

# Toxic / harmful language in output
_TOXIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(idiot|stupid|dumb|moron|incompetent)\b", re.I),
    re.compile(r"\b(kill|harm|destroy)\s+(yourself|the\s+farmer)", re.I),
]

# Sensitive information that should not appear in agent output
_SENSITIVE_OUTPUT_PATTERNS: Dict[str, re.Pattern] = {
    "api_key_like":  re.compile(r"[A-Za-z0-9_\-]{32,}"),   # very long tokens
    "bearer_token":  re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I),
    "private_key":   re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),
}

# Minimum useful response length (characters)
_MIN_OUTPUT_LENGTH = 50


def validate_output(response_text: str, agent_name: str = "agent") -> GuardrailResult:
    """
    Run all output guardrails on an AI-generated response.
    Returns GuardrailResult; if passed=False, do not display to user as-is.
    """
    if not response_text or not response_text.strip():
        _audit("OUTPUT", "empty_check", False, f"agent={agent_name}")
        return GuardrailResult(False, f"The {agent_name} returned an empty response. Please try again.")
    _audit("OUTPUT", "empty_check", True)

    # ── 2a. Minimum length ──────────────────────────────────────────────────
    if len(response_text.strip()) < _MIN_OUTPUT_LENGTH:
        _audit("OUTPUT", "length_check", False, f"agent={agent_name}, len={len(response_text)}")
        return GuardrailResult(False, f"The {agent_name} response is too short to be meaningful. Please try again.")
    _audit("OUTPUT", "length_check", True)

    # ── 2b. Hallucination detection ─────────────────────────────────────────
    for pattern in _HALLUCINATION_SIGNALS:
        if pattern.search(response_text):
            _audit("OUTPUT", "hallucination_detection", False, f"agent={agent_name}")
            return GuardrailResult(False, f"The {agent_name} response appears to contain uncertain or fabricated content. Please try again or consult an expert.")
    _audit("OUTPUT", "hallucination_detection", True)

    # ── 2c. Toxicity filtering ──────────────────────────────────────────────
    for pattern in _TOXIC_PATTERNS:
        if pattern.search(response_text):
            _audit("OUTPUT", "toxicity_filter", False, f"agent={agent_name}")
            return GuardrailResult(False, f"The {agent_name} response contains inappropriate content and has been blocked.")
    _audit("OUTPUT", "toxicity_filter", True)

    # ── 2d. Sensitive information filtering ─────────────────────────────────
    for label, pattern in _SENSITIVE_OUTPUT_PATTERNS.items():
        if label == "api_key_like":
            # Only flag very long non-URL token-looking strings
            matches = pattern.findall(response_text)
            suspicious = [m for m in matches if len(m) > 40 and not any(w in m.lower() for w in ["http", "www", "diagnos", "disease", "farmer"])]
            if suspicious:
                _audit("OUTPUT", "sensitive_data_filter", False, f"agent={agent_name}, type={label}")
                return GuardrailResult(False, f"The {agent_name} response may contain sensitive credential-like data and has been blocked for security.")
        else:
            if pattern.search(response_text):
                _audit("OUTPUT", "sensitive_data_filter", False, f"agent={agent_name}, type={label}")
                return GuardrailResult(False, f"The {agent_name} response contains sensitive information ({label}) and has been blocked.")
    _audit("OUTPUT", "sensitive_data_filter", True)

    # ── 2e. Response format validation (required section headings) ──────────
    if agent_name == "Pathologist Agent":
        missing = [h for h in _PATHOLOGIST_REQUIRED_HEADINGS if h not in response_text]
        if missing:
            _audit("OUTPUT", "format_validation", False, f"missing headings: {missing}")
            # Soft warning – don't block, just flag
            logging.warning("Pathologist output missing expected headings: %s", missing)
    elif agent_name == "Extension Officer Agent":
        missing = [h for h in _EXTENSION_REQUIRED_HEADINGS if h not in response_text]
        if missing:
            _audit("OUTPUT", "format_validation", False, f"missing headings: {missing}")
            logging.warning("Extension Officer output missing expected headings: %s", missing)
    _audit("OUTPUT", "format_validation", True)

    # ── 2f. Factual accuracy – disclaimer presence check ────────────────────
    disclaimer_present = any(phrase in response_text.lower() for phrase in [
        "not a confirmed", "preliminary", "ai-generated", "consult", "ai generated",
        "not a diagnosis", "professional advice",
    ])
    if agent_name == "Pathologist Agent" and not disclaimer_present:
        _audit("OUTPUT", "factual_accuracy", False, "no disclaimer in pathologist output")
        logging.warning("Pathologist output missing AI disclaimer – appending.")
        response_text = response_text + "\n\n---\n*⚠️ AI-generated preliminary assessment only. This is NOT a confirmed laboratory diagnosis. Please consult a certified plant pathologist.*"
    _audit("OUTPUT", "factual_accuracy", True)

    return GuardrailResult(True, "All output checks passed.", response_text)


# ===========================================================================
# 3. TOOL EXECUTION GUARDRAILS
# ===========================================================================

# Whitelist of allowed endpoints
_ALLOWED_ENDPOINTS: List[str] = [
    "https://api.replicate.com",
]

# Allowed HTTP methods per tool
_TOOL_PERMISSIONS: Dict[str, Dict[str, Any]] = {
    "replicate_inference": {
        "allowed_methods": ["POST"],
        "allowed_endpoints": ["https://api.replicate.com"],
        "max_tokens_per_call": 4096,
        "requires_auth": True,
    },
}


def validate_tool_call(tool_name: str, method: str, endpoint: str, payload: Optional[dict] = None) -> GuardrailResult:
    """
    Validate a tool/API call before it is executed.
    Checks: tool is registered, method is permitted, endpoint is whitelisted.
    """
    if tool_name not in _TOOL_PERMISSIONS:
        _audit("TOOL", "access_permission", False, f"tool={tool_name}")
        return GuardrailResult(False, f"Tool '{tool_name}' is not registered and cannot be called.")

    config = _TOOL_PERMISSIONS[tool_name]

    # Method check
    if method.upper() not in config["allowed_methods"]:
        _audit("TOOL", "method_check", False, f"tool={tool_name}, method={method}")
        return GuardrailResult(False, f"HTTP method '{method}' is not permitted for tool '{tool_name}'.")

    # Endpoint check
    allowed = config["allowed_endpoints"]
    if not any(endpoint.startswith(ep) for ep in allowed):
        _audit("TOOL", "endpoint_check", False, f"tool={tool_name}, endpoint={endpoint}")
        return GuardrailResult(False, f"Endpoint '{endpoint}' is not whitelisted for tool '{tool_name}'.")

    # Token budget check
    if payload and "max_tokens" in payload:
        max_allowed = config.get("max_tokens_per_call", 2000)
        if payload["max_tokens"] > max_allowed:
            _audit("TOOL", "token_budget", False, f"requested={payload['max_tokens']}, max={max_allowed}")
            return GuardrailResult(False, f"Requested max_tokens ({payload['max_tokens']}) exceeds the limit of {max_allowed}.")

    _audit("TOOL", "call_validation", True, f"tool={tool_name}")
    return GuardrailResult(True, "Tool call validated.", payload)


# ===========================================================================
# 4. DATA GUARDRAILS
# ===========================================================================

_PII_MASK_PATTERNS: Dict[str, Tuple[re.Pattern, str]] = {
    "email":       (_PII_PATTERNS["email"],       "[EMAIL REDACTED]"),
    "phone_intl":  (_PII_PATTERNS["phone_intl"],  "[PHONE REDACTED]"),
    "credit_card": (_PII_PATTERNS["credit_card"], "[CARD REDACTED]"),
    "aadhaar":     (_PII_PATTERNS["aadhaar"],     "[ID REDACTED]"),
    "pan_india":   (_PII_PATTERNS["pan_india"],   "[PAN REDACTED]"),
    "ssn_us":      (_PII_PATTERNS["ssn_us"],      "[SSN REDACTED]"),
    "ip_address":  (_PII_PATTERNS["ip_address"],  "[IP REDACTED]"),
}


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Replace PII in text with placeholders. Returns (masked_text, [types_found])."""
    masked = text
    found_types: List[str] = []
    for label, (pattern, placeholder) in _PII_MASK_PATTERNS.items():
        new_text = pattern.sub(placeholder, masked)
        if new_text != masked:
            found_types.append(label)
            masked = new_text
    return masked, found_types


def protect_data(data: Dict[str, Any], role: str = "user") -> GuardrailResult:
    """
    Apply data protection rules before passing data to agents.
    Masks PII from image metadata and symptom text.
    Role-based: 'user' gets masked data, 'admin' gets full data.
    """
    if role == "admin":
        _audit("DATA", "role_access", True, "admin bypass")
        return GuardrailResult(True, "Admin access – no masking applied.", data)

    protected = {}
    pii_summary: List[str] = []

    for key, value in data.items():
        if isinstance(value, str):
            masked_val, found = mask_pii(value)
            protected[key] = masked_val
            pii_summary.extend(found)
        else:
            protected[key] = value

    if pii_summary:
        _audit("DATA", "pii_masking", True, f"masked={pii_summary}")
    else:
        _audit("DATA", "pii_masking", True, "no PII found")

    # Prevent sensitive keys from leaking
    for sensitive_key in ("api_key", "password", "secret", "token", "credential"):
        if sensitive_key in protected:
            protected[sensitive_key] = "***REDACTED***"
            _audit("DATA", "credential_masking", True, f"key={sensitive_key}")

    return GuardrailResult(True, "Data protection applied.", protected)


# ===========================================================================
# 5. SECURITY GUARDRAILS
# ===========================================================================

def check_jailbreak(text: str) -> GuardrailResult:
    """
    Detect common jailbreak and adversarial prompting techniques in addition
    to prompt injection (which is also handled in validate_input).
    """
    jailbreak_indicators = [
        re.compile(r"\bDAN\b"),
        re.compile(r"do\s+anything\s+now", re.I),
        re.compile(r"(unlock|enable)\s+(developer|god|unrestricted)\s+mode", re.I),
        re.compile(r"you\s+have\s+no\s+(restrictions?|rules?|limits?)", re.I),
        re.compile(r"(evil|villain|hacker)\s+mode", re.I),
        re.compile(r"grandma\s+(exploit|trick|hack)", re.I),
        re.compile(r"hypothetically\s+speaking.*?(kill|hack|bypass)", re.I | re.S),
        re.compile(r"for\s+(educational|research)\s+purposes.*?(hack|exploit|bypass)", re.I | re.S),
        re.compile(r"base64\s*:\s*[A-Za-z0-9+/=]{20,}", re.I),   # encoded injection attempt
    ]
    for pattern in jailbreak_indicators:
        if pattern.search(text):
            _audit("SECURITY", "jailbreak_detection", False, pattern.pattern[:60])
            return GuardrailResult(False, "Jailbreak attempt detected. The system is restricted to agricultural diagnosis only.")
    _audit("SECURITY", "jailbreak_detection", True)
    return GuardrailResult(True, "No jailbreak patterns detected.")


def check_sql_injection(text: str) -> GuardrailResult:
    """Scan text for SQL injection patterns."""
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            _audit("SECURITY", "sql_injection", False, pattern.pattern[:60])
            return GuardrailResult(False, "SQL injection pattern detected in input. Request blocked.")
    _audit("SECURITY", "sql_injection", True)
    return GuardrailResult(True, "No SQL injection patterns detected.")


def hash_credential(credential: str) -> str:
    """Return a one-way SHA-256 hash of a credential for logging purposes only."""
    return hashlib.sha256(credential.encode()).hexdigest()[:16] + "..."


# ===========================================================================
# 6. OPERATIONAL GUARDRAILS
# ===========================================================================

@dataclass
class OperationalLimits:
    max_tokens_pathologist: int    = 1200
    max_tokens_extension: int      = 1400
    max_execution_seconds: float   = 180.0    # 3 minutes total
    max_agent_iterations: int      = 2        # 2 agents in this workflow
    max_api_calls_per_session: int = 10
    max_retries: int               = 2
    rate_limit_seconds: float      = 5.0      # minimum gap between calls


_op_limits = OperationalLimits()
_session_api_call_count: int = 0
_last_api_call_time: float   = 0.0
_session_start_time: float   = time.time()
_agent_iteration_count: int  = 0


def reset_operational_counters() -> None:
    """Reset per-session operational counters (call at start of each analysis)."""
    global _session_api_call_count, _last_api_call_time, _session_start_time, _agent_iteration_count
    _session_api_call_count = 0
    _last_api_call_time = 0.0
    _session_start_time = time.time()
    _agent_iteration_count = 0


def check_operational_limits(agent_name: str, max_tokens: int) -> GuardrailResult:
    """
    Enforce operational limits before each agent call.
    Checks: token budget, session API call count, rate limit, execution time,
    agent iteration count.
    """
    global _session_api_call_count, _last_api_call_time, _agent_iteration_count

    # ── 6a. Token budget ────────────────────────────────────────────────────
    if agent_name == "Pathologist Agent" and max_tokens > _op_limits.max_tokens_pathologist:
        _audit("OPERATIONAL", "token_budget", False, f"requested={max_tokens}")
        return GuardrailResult(False, f"Token limit exceeded for {agent_name} ({max_tokens} > {_op_limits.max_tokens_pathologist}).")

    if agent_name == "Extension Officer Agent" and max_tokens > _op_limits.max_tokens_extension:
        _audit("OPERATIONAL", "token_budget", False, f"requested={max_tokens}")
        return GuardrailResult(False, f"Token limit exceeded for {agent_name} ({max_tokens} > {_op_limits.max_tokens_extension}).")

    # ── 6b. API call count limit ────────────────────────────────────────────
    if _session_api_call_count >= _op_limits.max_api_calls_per_session:
        _audit("OPERATIONAL", "api_rate_limit", False, f"count={_session_api_call_count}")
        return GuardrailResult(False, f"Maximum API calls per session ({_op_limits.max_api_calls_per_session}) reached. Please start a new session.")

    # ── 6c. Rate limiting ───────────────────────────────────────────────────
    elapsed_since_last = time.time() - _last_api_call_time
    if _last_api_call_time > 0 and elapsed_since_last < _op_limits.rate_limit_seconds:
        wait_time = _op_limits.rate_limit_seconds - elapsed_since_last
        _audit("OPERATIONAL", "rate_limit", False, f"wait={wait_time:.1f}s")
        time.sleep(wait_time)   # enforce minimum gap

    # ── 6d. Max execution time ──────────────────────────────────────────────
    session_elapsed = time.time() - _session_start_time
    if session_elapsed > _op_limits.max_execution_seconds:
        _audit("OPERATIONAL", "max_execution_time", False, f"elapsed={session_elapsed:.1f}s")
        return GuardrailResult(False, "Session has exceeded the maximum allowed execution time. Please try again.")

    # ── 6e. Agent iteration count ───────────────────────────────────────────
    _agent_iteration_count += 1
    if _agent_iteration_count > _op_limits.max_agent_iterations:
        _audit("OPERATIONAL", "max_iterations", False, f"iterations={_agent_iteration_count}")
        return GuardrailResult(False, f"Maximum agent iterations ({_op_limits.max_agent_iterations}) exceeded. Possible infinite loop detected.")

    _session_api_call_count += 1
    _last_api_call_time = time.time()
    _audit("OPERATIONAL", "limits_check", True, f"agent={agent_name}, call_count={_session_api_call_count}")
    return GuardrailResult(True, "Operational limits OK.")


# ===========================================================================
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# ===========================================================================

@dataclass
class HumanApprovalRequest:
    action: str
    description: str
    risk_level: str                        # "low" | "medium" | "high" | "critical"
    requires_approval: bool = True
    approved: Optional[bool] = None
    reviewed_at: Optional[str] = None


def assess_human_review_need(diagnosis_text: str) -> HumanApprovalRequest:
    """
    Evaluate whether the diagnosis warrants a human-in-the-loop review step
    before the farmer acts on it.

    High-risk triggers: mentions of systemic pesticides, fungicides, or
    herbicides, or "critical" / "severe" disease confidence.
    """
    high_risk_keywords = [
        "systemic", "fungicide", "herbicide", "insecticide",
        "chemical treatment", "pesticide", "spray", "toxic",
        "severe", "critical", "outbreak", "quarantine",
        "notifiable", "regulated",
    ]
    text_lower = diagnosis_text.lower()
    matched = [kw for kw in high_risk_keywords if kw in text_lower]

    if matched:
        risk = "high"
        requires = True
        desc = f"Diagnosis includes recommendations involving chemicals or critical risk levels ({', '.join(matched[:3])}). Human expert review is recommended before action."
    elif "medium" in text_lower and "confidence" in text_lower:
        risk = "medium"
        requires = False
        desc = "Medium-confidence diagnosis. Consider consulting a local expert for confirmation."
    else:
        risk = "low"
        requires = False
        desc = "Low-risk preliminary diagnosis. Standard advisory applies."

    req = HumanApprovalRequest(
        action="crop_treatment_advisory",
        description=desc,
        risk_level=risk,
        requires_approval=requires,
    )
    _audit("HUMAN_LOOP", "risk_assessment", True, f"risk={risk}, triggers={matched[:3]}")
    return req


def record_human_decision(request: HumanApprovalRequest, approved: bool) -> None:
    """Record a human review decision against an approval request."""
    request.approved = approved
    request.reviewed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _audit("HUMAN_LOOP", "human_decision", approved, f"action={request.action}, risk={request.risk_level}")


# ===========================================================================
# 8. MULTI-AGENT GUARDRAILS
# ===========================================================================

_AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Pathologist Agent": {
        "allowed_input_types": ["image_context", "symptom_text"],
        "allowed_output_consumer": ["Extension Officer Agent", "report_builder"],
        "max_output_tokens": 1200,
    },
    "Extension Officer Agent": {
        "allowed_input_types": ["diagnosis_report"],
        "allowed_output_consumer": ["report_builder", "user"],
        "max_output_tokens": 1400,
    },
}

_TASK_SEQUENCE: List[str] = ["Pathologist Agent", "Extension Officer Agent"]
_completed_agents: List[str] = []


def reset_multi_agent_state() -> None:
    """Reset multi-agent state tracking for a new analysis run."""
    _completed_agents.clear()


def validate_agent_handoff(from_agent: str, to_agent: str, output_length: int) -> GuardrailResult:
    """
    Validate a handoff between agents.
    Checks: from_agent is registered, to_agent is an allowed consumer,
    task sequence is respected (no skipping / looping).
    """
    if from_agent not in _AGENT_REGISTRY:
        _audit("MULTI_AGENT", "agent_registry", False, f"unknown from_agent={from_agent}")
        return GuardrailResult(False, f"Agent '{from_agent}' is not a registered agent.")

    config = _AGENT_REGISTRY[from_agent]

    # Check authorised consumer
    if to_agent not in config["allowed_output_consumer"]:
        _audit("MULTI_AGENT", "access_control", False, f"{from_agent}→{to_agent}")
        return GuardrailResult(False, f"Agent '{from_agent}' is not authorised to pass output to '{to_agent}'.")

    # Enforce task sequence (prevent out-of-order or repeated execution)
    if from_agent in _completed_agents:
        _audit("MULTI_AGENT", "loop_prevention", False, f"{from_agent} already ran")
        return GuardrailResult(False, f"Agent '{from_agent}' has already completed its task. Possible agent loop detected.")

    expected_next = next(
        (a for a in _TASK_SEQUENCE if a not in _completed_agents), None
    )
    if expected_next and from_agent != expected_next:
        _audit("MULTI_AGENT", "sequence_validation", False, f"expected={expected_next}, got={from_agent}")
        return GuardrailResult(False, f"Task sequence violation: expected '{expected_next}' to run next, not '{from_agent}'.")

    # Output length check
    max_len = config["max_output_tokens"] * 4   # rough chars-per-token
    if output_length > max_len:
        _audit("MULTI_AGENT", "output_size", False, f"output_len={output_length}")
        return GuardrailResult(False, f"Agent '{from_agent}' output is excessively large ({output_length} chars). Potential runaway output.")

    _completed_agents.append(from_agent)
    _audit("MULTI_AGENT", "handoff_validated", True, f"{from_agent}→{to_agent}")
    return GuardrailResult(True, f"Handoff from {from_agent} to {to_agent} validated.")


def verify_cross_agent_output(source_agent: str, output_text: str) -> GuardrailResult:
    """
    Cross-verify the output of an agent before passing it to the next agent.
    Checks: minimum coherence, no injection in inter-agent messages.
    """
    # Check for injected instructions in agent output (adversarial chaining)
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(output_text):
            _audit("MULTI_AGENT", "cross_agent_injection", False, f"source={source_agent}")
            return GuardrailResult(False, f"Potential prompt injection detected in {source_agent} output. Handoff blocked.")

    if len(output_text.strip()) < 100:
        _audit("MULTI_AGENT", "output_coherence", False, f"source={source_agent}, len={len(output_text)}")
        return GuardrailResult(False, f"Output from {source_agent} is too short to be coherent. Pipeline halted.")

    _audit("MULTI_AGENT", "cross_agent_verification", True, f"source={source_agent}")
    return GuardrailResult(True, "Cross-agent output verification passed.")


# ===========================================================================
# 9. ETHICAL AND COMPLIANCE GUARDRAILS
# ===========================================================================

_BIAS_INDICATORS: List[re.Pattern] = [
    # Patterns suggesting advice differs based on region/social category unfairly
    re.compile(r"(poor|illiterate|uneducated)\s+farmer", re.I),
    re.compile(r"(only\s+for\s+(rich|wealthy|large))\s+farm", re.I),
    re.compile(r"(backward|primitive|third.world)\s+(farm|agriculture|region)", re.I),
]

_EXPLAINABILITY_KEYWORDS = [
    "because", "due to", "caused by", "indicates", "suggests", "based on",
    "evidence", "symptom", "colour", "pattern", "lesion",
]


def check_ethical_compliance(response_text: str, agent_name: str = "agent") -> GuardrailResult:
    """
    Run ethical and compliance checks on AI output.
    Checks: bias language, explainability, fairness indicators.
    """
    # ── 9a. Bias detection ──────────────────────────────────────────────────
    for pattern in _BIAS_INDICATORS:
        if pattern.search(response_text):
            _audit("ETHICS", "bias_detection", False, f"agent={agent_name}")
            return GuardrailResult(False, f"Biased or discriminatory language detected in {agent_name} output. Response blocked.")
    _audit("ETHICS", "bias_detection", True)

    # ── 9b. Explainability check ────────────────────────────────────────────
    text_lower = response_text.lower()
    has_explanation = any(kw in text_lower for kw in _EXPLAINABILITY_KEYWORDS)
    if not has_explanation and agent_name == "Pathologist Agent":
        _audit("ETHICS", "explainability", False, f"agent={agent_name}")
        logging.warning("Pathologist output lacks explanatory language – response may not be explainable.")
    else:
        _audit("ETHICS", "explainability", True)

    # ── 9c. Regulatory compliance – disclaimer check ─────────────────────────
    compliance_phrases = [
        "consult", "professional", "certified", "expert", "laboratory",
        "preliminary", "not a diagnosis",
    ]
    has_compliance = any(p in text_lower for p in compliance_phrases)
    if not has_compliance and agent_name == "Pathologist Agent":
        _audit("ETHICS", "regulatory_compliance", False, "no compliance disclaimer")
        # Append compliance notice rather than blocking
        response_text = (
            response_text
            + "\n\n---\n*⚠️ Regulatory Notice: This AI-generated assessment is preliminary "
            "and does not replace advice from a certified agricultural expert. Always comply "
            "with local agricultural regulations and pesticide usage laws.*"
        )
    _audit("ETHICS", "regulatory_compliance", True)

    # ── 9d. Fairness validation – response should be useful for all farmers ──
    fairness_issue = any(
        phrase in text_lower for phrase in [
            "not applicable to small farmers",
            "only for commercial farms",
            "not relevant for subsistence",
        ]
    )
    if fairness_issue:
        _audit("ETHICS", "fairness_validation", False, f"agent={agent_name}")
        return GuardrailResult(False, f"Fairness issue detected in {agent_name} output: response appears to exclude certain farmer categories.")
    _audit("ETHICS", "fairness_validation", True)

    return GuardrailResult(True, "Ethical and compliance checks passed.", response_text)


def get_compliance_report() -> Dict[str, Any]:
    """
    Generate a compliance summary from the audit trail.
    Returns counts of passed/failed checks per category.
    """
    summary: Dict[str, Dict[str, int]] = {}
    for record in _audit_log:
        cat = record["category"]
        if cat not in summary:
            summary[cat] = {"passed": 0, "failed": 0, "total": 0}
        summary[cat]["total"] += 1
        if record["passed"]:
            summary[cat]["passed"] += 1
        else:
            summary[cat]["failed"] += 1

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_checks": len(_audit_log),
        "overall_passed": sum(1 for r in _audit_log if r["passed"]),
        "overall_failed": sum(1 for r in _audit_log if not r["passed"]),
        "by_category": summary,
        "audit_log": _audit_log,
    }
