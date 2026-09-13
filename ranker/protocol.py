"""Strict wire protocol for the AI IME ranker.

The ranker is deliberately a *re-ranker*.  Candidate text is input only; it is
never present in a response.  This module is shared by the server and tests so
that the same checks are applied at both ends of the IPC boundary.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

MAX_CANDIDATES = 512
MAX_TEXT_BYTES = 32_768
MAX_LINE_BYTES = 2_097_152


class ProtocolError(ValueError):
    """A request or response violates the protocol contract."""


def _reject_duplicate_keys(pairs: List[Any]) -> Dict[str, Any]:
    """Reject duplicate object members instead of silently taking the last one."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    # NaN/Infinity are Python extensions, not JSON, and can bypass strict
    # numeric validation when they occur in an otherwise ignored field.
    raise ProtocolError(f"invalid JSON number: {value}")


def loads_strict(payload: str) -> Any:
    """Parse one JSON value with no duplicate keys or non-standard constants."""
    return json.loads(payload, object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=_reject_constant)


def _string(value: Any, name: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{name} must be a string")
    if len(value.encode("utf-8")) > max_bytes:
        raise ProtocolError(f"{name} is too large")
    return value


def validate_request(message: Any) -> Dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("request must be an object")
    required = {"request_id", "preceding_text", "read", "candidates"}
    allowed = required | {"following_text", "inference_trigger"}
    if not required.issubset(message) or not set(message).issubset(allowed):
        raise ProtocolError("request has unexpected or missing fields")

    request_id = _string(message["request_id"], "request_id", max_bytes=128)
    if not request_id:
        raise ProtocolError("request_id must not be empty")
    preceding = _string(message["preceding_text"], "preceding_text")
    following = _string(message["following_text"], "following_text") if "following_text" in message else ""
    inference_trigger = (
        _string(message["inference_trigger"], "inference_trigger", max_bytes=16)
        if "inference_trigger" in message
        else "interactive"
    )
    if inference_trigger not in {"explicit", "interactive", "prefetch"}:
        raise ProtocolError("inference_trigger must be explicit, interactive, or prefetch")
    reading = _string(message["read"], "read", max_bytes=512)
    candidates = message["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolError("candidates must be a non-empty array")
    if len(candidates) > MAX_CANDIDATES:
        raise ProtocolError("too many candidates")

    seen = set()
    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) != {"id", "text", "rank"}:
            raise ProtocolError("candidate fields must be exactly id,text,rank")
        cid = _string(item["id"], "candidate.id", max_bytes=128)
        text = _string(item["text"], "candidate.text", max_bytes=4096)
        rank = item["rank"]
        if not cid or cid in seen:
            raise ProtocolError("candidate ids must be unique and non-empty")
        if (isinstance(rank, bool) or not isinstance(rank, int) or
                rank != len(normalized) + 1):
            raise ProtocolError("candidate.rank must be contiguous starting at 1")
        seen.add(cid)
        normalized.append({"id": cid, "text": text, "rank": rank})
    return {
        "request_id": request_id,
        "preceding_text": preceding,
        "following_text": following,
        "inference_trigger": inference_trigger,
        "read": reading,
        "candidates": normalized,
    }


def validate_response(message: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a response only re-ranks IDs from *request*.

    The response schema intentionally has no ``text`` field.  Unknown IDs,
    duplicate IDs, non-finite scores, gaps, and any extra fields are rejected.
    """
    if not isinstance(message, dict):
        raise ProtocolError("response must be an object")
    if set(message) != {"request_id", "candidates"}:
        raise ProtocolError("response has unexpected or missing fields")
    if message["request_id"] != request["request_id"]:
        raise ProtocolError("request_id mismatch")
    items = message["candidates"]
    if not isinstance(items, list) or len(items) != len(request["candidates"]):
        raise ProtocolError("response must contain every input candidate")
    allowed = {x["id"] for x in request["candidates"]}
    seen = set()
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "score", "rank"}:
            raise ProtocolError("response candidate fields must be exactly id,score,rank")
        cid = _string(item["id"], "response.id", max_bytes=128)
        score = item["score"]
        rank = item["rank"]
        if cid not in allowed or cid in seen:
            raise ProtocolError("response contains unknown or duplicate id")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ProtocolError("response.score must be finite")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != len(seen) + 1:
            raise ProtocolError("response ranks must be contiguous starting at 1")
        seen.add(cid)
        normalized.append({"id": cid, "score": float(score), "rank": rank})
    if seen != allowed:
        raise ProtocolError("response omitted an input id")
    return {"request_id": request["request_id"], "candidates": normalized}
