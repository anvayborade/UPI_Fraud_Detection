from __future__ import annotations

import hashlib
import re

from src.intelligence.complaint_schema import ScamIntelligence
from src.intelligence.llm_client import ComplaintLLMClient

VPA_PATTERN = re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b")
MOBILE_PATTERN = re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)")


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redact_identifiers(text: str) -> tuple[str, list[str], list[str]]:
    vpas = VPA_PATTERN.findall(text)
    mobiles = MOBILE_PATTERN.findall(text)
    redacted = text
    for vpa in vpas:
        redacted = redacted.replace(vpa, f"<VPA:{_hash_identifier(vpa.lower())}>")
    for mobile in mobiles:
        redacted = redacted.replace(mobile, f"<MOBILE:{_hash_identifier(mobile)}>")
    return redacted, [_hash_identifier(v.lower()) for v in vpas], [_hash_identifier(m) for m in mobiles]


def fallback_extract(text: str) -> ScamIntelligence:
    lowered = text.lower()
    refund = any(word in lowered for word in ["refund", "cashback", "receive money"])
    qr = "qr" in lowered or "scan" in lowered
    collect = "collect request" in lowered or "request money" in lowered
    remote = any(word in lowered for word in ["anydesk", "teamviewer", "screen share", "remote access"])
    impersonation = any(word in lowered for word in ["police", "bank officer", "customer care", "courier", "customs"])
    urgency = any(word in lowered for word in ["urgent", "immediately", "now", "account blocked", "arrest"])

    if remote:
        scam_type = "REMOTE_ACCESS"
    elif collect:
        scam_type = "COLLECT_REQUEST"
    elif refund and qr:
        scam_type = "FAKE_REFUND"
    elif qr:
        scam_type = "QR_DECEPTION"
    elif impersonation:
        scam_type = "IMPERSONATION"
    else:
        scam_type = "UNKNOWN"
    confidence = 0.55 + 0.08 * sum([refund, qr, collect, remote, impersonation, urgency])
    return ScamIntelligence(
        scam_type=scam_type,
        urgency_language=urgency,
        refund_claim=refund,
        remote_access_mentioned=remote,
        collect_request_mentioned=collect,
        qr_code_mentioned=qr,
        confidence=min(confidence, 0.95),
    )


def extract_complaint(text: str, use_llm: bool = True) -> ScamIntelligence:
    redacted, vpas, mobiles = redact_identifiers(text)
    result: ScamIntelligence
    if use_llm:
        try:
            result = ComplaintLLMClient().extract(redacted)
        except Exception:
            result = fallback_extract(redacted)
    else:
        result = fallback_extract(redacted)
    result.referenced_vpas = sorted(set(result.referenced_vpas + vpas))
    result.referenced_mobile_hashes = sorted(set(result.referenced_mobile_hashes + mobiles))
    return result
