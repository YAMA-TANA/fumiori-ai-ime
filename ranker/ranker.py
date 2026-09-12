"""Resident, candidate-only AI IME ranker.

Run with ``--stdio`` for a portable E2E harness, or ``--pipe`` on Windows for
the production-shaped Named Pipe transport.  ``--backend rule`` uses the
deterministic acceptance-test scorer; ``--backend qwen`` uses the optional
Qwen3-Reranker forward-only yes/no-logit scorer.  Both backends only reorder
candidate IDs already supplied by Mozc and share the same protocol and
fallback behavior.
"""

from __future__ import annotations

import argparse
import ctypes
from collections import OrderedDict
from copy import deepcopy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from product_settings import load_settings
try:
    from .lexicon import contextual_candidate_bonus
except ImportError:
    from lexicon import contextual_candidate_bonus

try:  # When run as ``python -m ranker.ranker``.
    from .protocol import (
        MAX_LINE_BYTES,
        ProtocolError,
        loads_strict,
        validate_batch_request,
        validate_batch_response,
        validate_request,
        validate_response,
    )
    from .loading_ui import LoadingIndicator
except ImportError:  # When run as ``python ranker/ranker.py``.
    from protocol import (
        MAX_LINE_BYTES,
        ProtocolError,
        loads_strict,
        validate_batch_request,
        validate_batch_response,
        validate_request,
        validate_response,
    )
    from loading_ui import LoadingIndicator

QwenReranker = None
QwenDependencyError = RuntimeError

LOG = logging.getLogger("ai_ime_ranker")
WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"
LOW_INTEGRITY_PIPE_SDDL = (
    "D:P(A;;GA;;;SY)(A;;GA;;;OW)"
    "S:(ML;;NW;;;LW)"
)


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("security_descriptor", ctypes.c_void_p),
        ("inherit_handle", ctypes.c_int),
    ]


def _low_integrity_pipe_security() -> tuple[_SecurityAttributes, ctypes.c_void_p]:
    """Build same-owner pipe security that accepts Mozc's low-integrity token."""
    if sys.platform != "win32":
        raise OSError("Windows pipe security is available only on Windows")
    advapi32 = ctypes.windll.advapi32
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = ctypes.c_int
    descriptor = ctypes.c_void_p()
    descriptor_size = ctypes.c_uint32()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        LOW_INTEGRITY_PIPE_SDDL,
        1,  # SDDL_REVISION_1
        ctypes.byref(descriptor),
        ctypes.byref(descriptor_size),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    attributes = _SecurityAttributes(
        ctypes.sizeof(_SecurityAttributes), descriptor, False
    )
    return attributes, descriptor


def normalize_windows_pipe_name(pipe_name: str) -> str:
    """Return one canonical local Windows named-pipe path.

    The former implementation compared against a raw string containing two
    backslashes after ``pipe``.  A valid full path therefore failed the check
    and became ``\\.\pipe\\.\pipe\...``.  Mozc always connects to the
    canonical path, so that bug made every real IME request silently fall back.
    """
    name = str(pipe_name).strip()
    if not name:
        raise ValueError("pipe name must not be empty")
    while name.startswith(WINDOWS_PIPE_PREFIX):
        name = name[len(WINDOWS_PIPE_PREFIX):]
    name = name.lstrip("\\")
    # Repair names produced by the historical duplicate-prefix bug too.
    while name.startswith(".\\pipe\\"):
        name = name[len(".\\pipe\\"):].lstrip("\\")
    if not name or name in {".", ".."}:
        raise ValueError("pipe name must contain a local name")
    return WINDOWS_PIPE_PREFIX + name


def summarize_reorder(request: Dict[str, Any], response: Dict[str, Any]) -> tuple[bool, int]:
    """Return whether order changed and the original rank of the new top ID."""
    original_ids = [str(item["id"]) for item in request["candidates"]]
    ranked_ids = [str(item["id"]) for item in response["candidates"]]
    if not original_ids or len(original_ids) != len(ranked_ids):
        raise ValueError("candidate ID lists must be non-empty and equally sized")
    return ranked_ids != original_ids, original_ids.index(ranked_ids[0]) + 1


class RuntimeStatus:
    """Small, privacy-preserving status file consumed by the tray UI."""

    def __init__(self, path: Optional[str], backend: str, model: str, pipe: str):
        self.path = Path(path) if path else None
        self.data = {
            "schema": 1,
            "state": "loading",
            "pid": os.getpid(),
            "backend": backend,
            "model": model,
            "pipe": pipe,
            "requests": 0,
            "ime_requests": 0,
            "ime_reordered_requests": 0,
            "ime_top_changed_requests": 0,
            "last_ime_order_changed": None,
            "last_promoted_from_rank": None,
            "last_context_chars": None,
            "last_read_chars": None,
            "last_candidate_count": None,
            "last_segment_count": None,
            "last_request_at": None,
            "last_latency_ms": None,
        }

    def write(self, **updates: Any) -> None:
        if self.path is None:
            return
        try:
            self.data.update(updates)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, self.path)
        except OSError as exc:
            # Diagnostics must never break the IME conversion path.
            LOG.warning("could not update runtime status: %s", exc)


class RuleBasedRanker:
    """Score only supplied candidates; never creates a candidate string."""

    def __init__(self, delay_ms: float = 0.0):
        self.delay_ms = max(0.0, delay_ms)

    @staticmethod
    def _context_score(context: str, candidate_text: str) -> float:
        # These are intentionally exact input candidates, not generated text.
        contextual_bonus = contextual_candidate_bonus(context, candidate_text)
        if contextual_bonus:
            return contextual_bonus
        if "大きい" in context:
            if candidate_text == "鼻":
                return 100.0
            if candidate_text in {"花", "華"}:
                return 8.0
        if "電子工学" in context or "計算" in context:
            if candidate_text in {"機械", "計算機械"}:
                return 100.0
            if candidate_text in {"機会", "計算機会", "器械", "計算器械"}:
                return 8.0
        return 0.0

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request = validate_request(request)
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        prefix = request["preceding_text"]
        suffix = request.get("following_text", "")
        context = f"{prefix} {request['read']} {suffix}".strip()
        scored = []
        for candidate in request["candidates"]:
            # Original rank is a stable tie-breaker and a safe baseline.
            score = float(len(request["candidates"]) - candidate["rank"])
            score += self._context_score(context, candidate["text"])
            scored.append((score, -candidate["rank"], candidate["id"]))
        scored.sort(reverse=True)
        result = [
            {"id": cid, "score": score, "rank": index}
            for index, (score, _old_rank, cid) in enumerate(scored, start=1)
        ]
        # There is deliberately no candidate text in this return value.
        return {"request_id": request["request_id"], "candidates": result}


class InteractiveBurstGuard:
    """Run explicit conversion now; debounce legacy interactive requests."""

    def __init__(
        self,
        delegate: Any,
        *,
        settle_seconds: float = 0.5,
        forget_seconds: float = 2.0,
    ) -> None:
        self._delegate = delegate
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.forget_seconds = max(self.settle_seconds, float(forget_seconds))
        self._first_seen: Dict[tuple[Any, ...], float] = {}
        self._ranked: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _key(request: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            request.get("inference_trigger", "interactive"),
            request["preceding_text"],
            request.get("following_text", ""),
            request["read"],
            tuple(
                (candidate["id"], candidate["text"], candidate["rank"])
                for candidate in request["candidates"]
            ),
        )

    @staticmethod
    def _original_order(request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "request_id": request["request_id"],
            "candidates": [
                {
                    "id": candidate["id"],
                    "score": float(-candidate["rank"]),
                    "rank": candidate["rank"],
                }
                for candidate in request["candidates"]
            ],
        }

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request = validate_request(request)
        # New Mozc clients label a Space/Convert operation explicitly. It must
        # not wait for the compatibility debounce used by older clients.
        if request["inference_trigger"] == "explicit":
            return self._delegate.rank(request)

        now = time.perf_counter()
        key = self._key(request)
        for old_key, seen_at in tuple(self._first_seen.items()):
            if now - seen_at > self.forget_seconds:
                self._first_seen.pop(old_key, None)
                self._ranked.pop(old_key, None)

        if key in self._ranked:
            return {
                "request_id": request["request_id"],
                "candidates": self._ranked[key],
            }

        first_seen = self._first_seen.get(key)
        if first_seen is None:
            self._first_seen[key] = now
            return self._original_order(request)
        if now - first_seen < self.settle_seconds:
            return self._original_order(request)

        response = self._delegate.rank(request)
        self._ranked[key] = response["candidates"]
        return response


def _compact_batch_fallback(
    request: Dict[str, Any], ranker: Any
) -> Dict[str, Any]:
    """Adapt a legacy backend to the compact batch response."""
    results = []
    for segment in request["segments"]:
        single_request = {
            "request_id": f"{request['request_id']}-{segment['id']}",
            "inference_trigger": request["inference_trigger"],
            "preceding_text": segment["preceding_text"],
            "following_text": segment["following_text"],
            "read": segment["read"],
            "candidates": segment["candidates"],
        }
        response = validate_response(ranker.rank(single_request), single_request)
        ranked = response["candidates"]
        winner = ranked[0]
        values = [float(item["score"]) for item in ranked]
        maximum = max(values)
        denominator = sum(math.exp(value - maximum) for value in values)
        confidence = math.exp(float(winner["score"]) - maximum) / denominator
        results.append({
            "id": segment["id"],
            "winner_id": winner["id"],
            "confidence": confidence,
        })
    return {"request_id": request["request_id"], "segments": results}


class ResponseCache:
    """Short-lived cache for repeated explicit conversion requests."""

    def __init__(
        self,
        delegate: Any,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 3.0,
    ) -> None:
        self._delegate = delegate
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._entries: OrderedDict[str, tuple[float, Dict[str, Any]]] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _key(request: Dict[str, Any]) -> str:
        value = dict(request)
        value.pop("request_id", None)
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def _get_or_compute(self, request: Dict[str, Any], compute: Any) -> Dict[str, Any]:
        key = self._key(request)
        now = time.perf_counter()
        entry = self._entries.get(key)
        if entry is not None:
            stored_at, response = entry
            if now - stored_at <= self.ttl_seconds:
                self._entries.move_to_end(key)
                cached = deepcopy(response)
                cached["request_id"] = request["request_id"]
                return cached
            self._entries.pop(key, None)

        response = compute(request)
        if response is not None:
            self._entries[key] = (time.perf_counter(), deepcopy(response))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return response

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        normalized = validate_request(request)
        return self._get_or_compute(normalized, self._delegate.rank)

    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        normalized = validate_batch_request(request)
        if hasattr(self._delegate, "rank_batch"):
            compute = self._delegate.rank_batch
        else:
            compute = lambda value: _compact_batch_fallback(value, self._delegate)
        return self._get_or_compute(normalized, compute)


def summarize_batch_reorder(
    request: Dict[str, Any], response: Dict[str, Any]
) -> tuple[int, int]:
    """Return changed-segment and non-baseline-winner counts."""
    winners = {item["id"]: item["winner_id"] for item in response["segments"]}
    changed = 0
    promoted = 0
    for segment in request["segments"]:
        winner_id = winners[segment["id"]]
        original_top = segment["candidates"][0]["id"]
        if winner_id != original_top:
            changed += 1
            promoted += 1
    return changed, promoted


def process_line(line: bytes, ranker: Any) -> Optional[bytes]:
    try:
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError("JSON line is too large")
        request = loads_strict(line.decode("utf-8"))
        if isinstance(request, dict) and "segments" in request:
            req_val = validate_batch_request(request)
            if hasattr(ranker, "rank_batch"):
                response = ranker.rank_batch(req_val)
            else:
                response = _compact_batch_fallback(req_val, ranker)
            clean_response = validate_batch_response(response, req_val)
            return (
                json.dumps(clean_response, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        req_val = validate_request(request)
        if (not req_val["preceding_text"] and
                not req_val["following_text"] and
                req_val["inference_trigger"] != "explicit"):
            # A legacy context-free interactive request has no information
            # with which to improve Mozc's dictionary order.  Explicit Mozc
            # conversion is allowed through because a resized compound can
            # still be judged by whole-word naturalness, and a restored split
            # supplies neighboring segments as prefix/suffix context.
            response = {
                "request_id": req_val["request_id"],
                "candidates": [
                    {
                        "id": candidate["id"],
                        "score": float(-candidate["rank"]),
                        "rank": candidate["rank"],
                    }
                    for candidate in req_val["candidates"]
                ],
            }
        else:
            response = ranker.rank(req_val)
        # Normalize and validate our own response before it crosses the trust boundary.
        clean_response = {
            "request_id": response["request_id"],
            "candidates": [
                {"id": str(c["id"]), "score": float(c["score"]), "rank": int(c["rank"])}
                for c in response["candidates"]
            ],
        }
        validate_response(clean_response, req_val)
        return (json.dumps(clean_response, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, ProtocolError, TypeError, ValueError) as exc:
        LOG.warning("dropping invalid request: %s", exc)
        return None
    except Exception as exc:
        # Backend failures must not kill the resident process.  Returning no
        # line makes the C++ client fail closed and preserve Mozc's original
        # ordering; the next request can still be served after a transient
        # model/torch/tokenizer error.
        LOG.exception("ranker backend failed; forcing client fallback: %s", exc)
        return None


def serve_stdio(ranker: Any, once: bool = False) -> int:
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        started = time.perf_counter()
        output = process_line(line, ranker)
        if output is not None:
            try:
                sys.stdout.buffer.write(output)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                return 0
            LOG.info("ranked in %.1f ms", (time.perf_counter() - started) * 1000)
        if once:
            break
    return 0


def _windows_pipe_server(pipe_name: str, ranker: Any, show_ui: bool = True,
                         runtime_status: Optional[RuntimeStatus] = None) -> int:
    if sys.platform != "win32":
        LOG.error("--pipe is available only on Windows")
        return 2
    pipe_name = normalize_windows_pipe_name(pipe_name)
    k32 = ctypes.windll.kernel32
    PIPE_ACCESS_DUPLEX = 0x00000003
    PIPE_TYPE_BYTE = 0x00000000
    PIPE_READMODE_BYTE = 0x00000000
    PIPE_WAIT = 0x00000000
    # Do not expose the local ranker to remote SMB clients.  The IME and
    # ranker are expected to be on the same Windows host.
    PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    k32.CreateNamedPipeW.restype = ctypes.c_void_p
    k32.CreateNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_uint32, ctypes.c_void_p]
    k32.ConnectNamedPipe.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                              ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    k32.DisconnectNamedPipe.argtypes = [ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.LocalFree.argtypes = [ctypes.c_void_p]
    k32.LocalFree.restype = ctypes.c_void_p

    try:
        pipe_security, security_descriptor = _low_integrity_pipe_security()
    except OSError as exc:
        LOG.error("Could not create local pipe security descriptor: %s", exc)
        return 2

    indicator = LoadingIndicator(text="AI変換中…", enabled=show_ui)
    indicator.start()

    LOG.info("listening on %s (loading_ui=%s)", pipe_name, show_ui)
    if runtime_status is not None:
        runtime_status.write(state="on", pipe=pipe_name, ready_at=time.time())
    try:
        while True:
            handle = k32.CreateNamedPipeW(
                pipe_name, PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                1, 65536, 65536, 0, ctypes.byref(pipe_security))
            if not handle or handle == INVALID_HANDLE_VALUE:
                LOG.error("CreateNamedPipeW failed: %s", ctypes.get_last_error())
                return 2
            connected = k32.ConnectNamedPipe(handle, None)
            if not connected and ctypes.get_last_error() != 535:  # ERROR_PIPE_CONNECTED
                LOG.warning("ConnectNamedPipe failed: %s", ctypes.get_last_error())
                k32.CloseHandle(handle)
                continue
            data = bytearray()
            try:
                while len(data) < MAX_LINE_BYTES:
                    buf = ctypes.create_string_buffer(4096)
                    read = ctypes.c_uint32()
                    ok = k32.ReadFile(handle, buf, len(buf), ctypes.byref(read), None)
                    if not ok or read.value == 0:
                        break
                    data.extend(buf.raw[:read.value])
                    if b"\n" in data:
                        break
                newline = data.find(b"\n")
                if newline >= 0:
                    request_id = ""
                    observed: Dict[str, Any] = {}
                    try:
                        observed = loads_strict(bytes(data[:newline]).decode("utf-8"))
                        request_id = str(observed.get("request_id", ""))
                    except Exception:
                        pass
                    with indicator.active():
                        started = time.perf_counter()
                        output = process_line(bytes(data[: newline + 1]), ranker)
                    if output is not None:
                        written = ctypes.c_uint32()
                        k32.WriteFile(handle, output, len(output), ctypes.byref(written), None)
                        k32.FlushFileBuffers(handle)
                        LOG.info("ranked pipe request in %.3f ms",
                                 (time.perf_counter() - started) * 1000.0)
                        if runtime_status is not None:
                            latency_ms = (time.perf_counter() - started) * 1000.0
                            is_ime = request_id.startswith("mozc-")
                            updates: Dict[str, Any] = dict(
                                requests=int(runtime_status.data["requests"]) + 1,
                                ime_requests=(
                                    int(runtime_status.data["ime_requests"]) + 1
                                    if is_ime
                                    else int(runtime_status.data["ime_requests"])
                                ),
                                last_request_at=time.time(),
                                last_latency_ms=round(latency_ms, 3),
                                last_source="mozc" if is_ime else "probe",
                            )
                            if is_ime:
                                response = loads_strict(output.decode("utf-8"))
                                if isinstance(observed.get("segments"), list):
                                    changed_count, promoted_count = summarize_batch_reorder(
                                        observed, response
                                    )
                                    updates.update(
                                        ime_reordered_requests=(
                                            int(runtime_status.data["ime_reordered_requests"])
                                            + (1 if changed_count else 0)
                                        ),
                                        ime_top_changed_requests=(
                                            int(runtime_status.data["ime_top_changed_requests"])
                                            + promoted_count
                                        ),
                                        last_ime_order_changed=bool(changed_count),
                                        last_promoted_from_rank=None,
                                        last_context_chars=sum(
                                            len(str(item.get("preceding_text", "")))
                                            for item in observed["segments"]
                                        ),
                                        last_read_chars=sum(
                                            len(str(item.get("read", "")))
                                            for item in observed["segments"]
                                        ),
                                        last_candidate_count=sum(
                                            len(item.get("candidates", []))
                                            for item in observed["segments"]
                                        ),
                                        last_segment_count=len(observed["segments"]),
                                    )
                                    LOG.info(
                                        "IME batch reorder changed_segments=%s promoted_segments=%s segments=%s",
                                        changed_count, promoted_count,
                                        len(observed["segments"]),
                                    )
                                else:
                                    changed, promoted_from = summarize_reorder(observed, response)
                                    updates.update(
                                        ime_reordered_requests=(
                                            int(runtime_status.data["ime_reordered_requests"])
                                            + (1 if changed else 0)
                                        ),
                                        ime_top_changed_requests=(
                                            int(runtime_status.data["ime_top_changed_requests"])
                                            + (1 if promoted_from > 1 else 0)
                                        ),
                                        last_ime_order_changed=changed,
                                        last_promoted_from_rank=promoted_from,
                                        last_context_chars=len(str(observed.get("preceding_text", ""))),
                                        last_read_chars=len(str(observed.get("read", ""))),
                                        last_candidate_count=len(observed.get("candidates", [])),
                                        last_segment_count=1,
                                    )
                                    LOG.info(
                                        "IME reorder changed=%s top_from=%s context_chars=%s read_chars=%s",
                                        changed, promoted_from,
                                        updates["last_context_chars"], updates["last_read_chars"],
                                    )
                            runtime_status.write(**updates)
            finally:
                k32.DisconnectNamedPipe(handle)
                k32.CloseHandle(handle)
    finally:
        indicator.stop()
        k32.LocalFree(security_descriptor)
        if runtime_status is not None:
            runtime_status.write(state="off", stopped_at=time.time())


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stdio", action="store_true", help="JSON Lines over stdin/stdout")
    mode.add_argument("--once", action="store_true", help="process one stdio request and exit")
    mode.add_argument("--pipe", metavar="NAME", help="Windows Named Pipe name")
    parser.add_argument(
        "--backend", choices=("rule", "qwen", "ruri", "onnx"), default="onnx",
        help="candidate scoring backend (default: self-contained ONNX Ruri)",
    )
    parser.add_argument(
        "--model-name", default="cl-nagoya/ruri-v3-reranker-310m",
        help="Hugging Face model ID/path for --backend ruri or qwen",
    )
    parser.add_argument(
        "--ensemble-model", action="append", default=None,
        help=(
            "ONNX model path; repeat for a weighted ensemble "
            "(two paths use the calibrated LoRA3/LoRA6 25/75 default)"
        ),
    )
    parser.add_argument(
        "--query-variant", choices=("current", "short", "none"), default="current",
        help="ONNX query template used for ranking (default: current)",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="input token limit (default: 512)",
    )
    parser.add_argument(
        "--prompt-variant", choices=("baseline", "dual"), default="baseline",
        help="Qwen query/document layout (dual fuses word and completed-sentence views)",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto",
        help="Qwen parameter dtype; auto uses fp16 on CUDA and fp32 on CPU",
    )
    parser.add_argument(
        "--disable-warmup", action="store_true",
        help="skip the pre-listen Qwen CUDA/kernel warmup (pipe mode only)",
    )
    parser.add_argument(
        "--no-ui", action="store_true",
        help="disable the floating 'AI変換中…' loading indicator overlay",
    )
    parser.add_argument(
        "--status-file", default=None,
        help="write runtime/backend/request counters for the local tray UI",
    )
    parser.add_argument(
        "--settings-file", default=None,
        help="read local Yamatana AI IME product settings",
    )
    parser.add_argument("--delay-ms", type=float, default=0.0, help="test-only artificial delay")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.backend == "qwen":
            try:
                from .qwen_ranker import QwenReranker as qwen_reranker_class
            except ImportError:
                from ranker.qwen_ranker import QwenReranker as qwen_reranker_class
            load_started = time.perf_counter()
            ranker = qwen_reranker_class(
                args.model_name, device=args.device, max_length=args.max_length,
                prompt_variant=args.prompt_variant, dtype=args.dtype,
            )
            LOG.info(
                "Qwen model loaded in %.1f ms (device=%s dtype=%s prompt=%s)",
                (time.perf_counter() - load_started) * 1000.0,
                ranker.device, ranker.dtype, ranker.prompt_variant,
            )
            if args.pipe and not args.disable_warmup:
                warmup_started = time.perf_counter()
                ranker.warmup()
                LOG.info(
                    "Qwen warmup completed in %.1f ms; pipe is ready",
                    (time.perf_counter() - warmup_started) * 1000.0,
                )
        elif args.backend == "onnx":
            try:
                from .onnx_ranker import OnnxRuriReranker
            except ImportError:
                from ranker.onnx_ranker import OnnxRuriReranker
            load_started = time.perf_counter()
            product_settings = load_settings(args.settings_file)
            ranker = OnnxRuriReranker(
                settings=product_settings,
                model_paths=args.ensemble_model,
                query_variant=args.query_variant,
            )
            LOG.info(
                "ONNX Ruri model loaded in %.1f ms (device=%s)",
                (time.perf_counter() - load_started) * 1000.0,
                ranker.device,
            )
        elif args.backend == "ruri":
            try:
                from .ruri_ranker import RuriReranker
            except ImportError:
                try:
                    from ranker.ruri_ranker import RuriReranker
                except ImportError:
                    from ruri_ranker import RuriReranker
            load_started = time.perf_counter()
            product_settings = load_settings(args.settings_file)
            ranker = RuriReranker(
                enable_lexical_grounding=True,
                device=args.device,
                settings=product_settings,
            )
            LOG.info(
                "Ruri model loaded in %.1f ms (device=%s)",
                (time.perf_counter() - load_started) * 1000.0,
                ranker.device,
            )
        else:
            ranker = RuleBasedRanker(args.delay_ms)
    except Exception as exc:
        # A failed ranker process is explicit. Mozc treats process failure as
        # an immediate fallback to its original candidate order.
        LOG.error("could not start %s backend: %s", args.backend, exc)
        return 2
    if args.pipe:
        if args.backend in {"ruri", "onnx"}:
            # A current Mozc process marks Space/Convert as explicit. Requests
            # from older binaries have no marker and are treated as live input:
            # return Mozc order immediately until the same request is stable
            # for 500 ms, so typing never starts model inference.
            ranker = InteractiveBurstGuard(ranker, settle_seconds=0.5)
        ranker = ResponseCache(ranker)
        model_paths = getattr(ranker, "model_paths", None)
        model = (
            ",".join(str(path) for path in model_paths)
            if model_paths
            else getattr(ranker, "model_path", args.model_name if args.backend != "rule" else "rule")
        )
        status = RuntimeStatus(
            args.status_file, args.backend, str(model), normalize_windows_pipe_name(args.pipe)
        )
        if args.backend in {"ruri", "onnx"}:
            status.write(
                compute_device=str(getattr(ranker, "device", "unknown")),
                document_domain=str(getattr(ranker, "document_domain", "general")),
                context_enabled=bool(getattr(ranker, "context_enabled", True)),
                context_chars=int(getattr(ranker, "context_chars", 0)),
                lexical_grounding=bool(getattr(ranker, "enable_lexical_grounding", False)),
            )
        return _windows_pipe_server(
            args.pipe, ranker, show_ui=not args.no_ui, runtime_status=status
        )
    return serve_stdio(ranker, once=args.once or not args.stdio)


if __name__ == "__main__":
    raise SystemExit(main())
