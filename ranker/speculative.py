"""Typing-time speculative rerank cache for Yamatana AI IME.

Prefetch requests never block the IME on neural inference.  They replace the
latest queued request, wait for a short typing-settle window, and score on one
background worker.  A later explicit conversion reuses an exact normalized
cache hit; on a miss it falls back to a normal synchronous rank.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict

try:
    from .protocol import validate_request
except ImportError:
    from protocol import validate_request


class SpeculativeRanker:
    def __init__(
        self,
        delegate: Any,
        *,
        settle_seconds: float = 0.04,
        cache_ttl_seconds: float = 2.0,
        cache_size: int = 64,
    ) -> None:
        self._delegate = delegate
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.cache_ttl_seconds = max(0.1, float(cache_ttl_seconds))
        self.cache_size = max(1, int(cache_size))
        self._condition = threading.Condition()
        self._inference_lock = threading.Lock()
        self._pending: Dict[str, Any] | None = None
        self._pending_at = 0.0
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, list[Dict[str, Any]]]] = OrderedDict()
        self._stopped = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="YamatanaSpeculativeRanker",
            daemon=True,
        )
        self._worker.start()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

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

    def _context_limit(self) -> int:
        try:
            if not bool(getattr(self._delegate, "context_enabled", True)):
                return 0
            return max(0, int(getattr(self._delegate, "context_chars", 64)))
        except (TypeError, ValueError):
            return 64

    def _key(self, request: Dict[str, Any]) -> tuple[Any, ...]:
        limit = self._context_limit()
        prefix = request["preceding_text"][-limit:] if limit else ""
        suffix = request.get("following_text", "")[:limit] if limit else ""
        return (
            prefix,
            suffix,
            request["read"],
            tuple(
                (candidate["id"], candidate["text"], candidate["rank"])
                for candidate in request["candidates"]
            ),
        )

    def _prune_locked(self, now: float) -> None:
        expired = [
            key for key, (stored_at, _value) in self._cache.items()
            if now - stored_at > self.cache_ttl_seconds
        ]
        for key in expired:
            self._cache.pop(key, None)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _lookup(self, request: Dict[str, Any]) -> Dict[str, Any] | None:
        key = self._key(request)
        now = time.monotonic()
        with self._condition:
            self._prune_locked(now)
            item = self._cache.get(key)
            if item is None:
                return None
            _stored_at, candidates = item
            self._cache.move_to_end(key)
            return {
                "request_id": request["request_id"],
                "candidates": [dict(candidate) for candidate in candidates],
            }

    def _store(self, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        key = self._key(request)
        candidates = [dict(candidate) for candidate in response["candidates"]]
        now = time.monotonic()
        with self._condition:
            self._cache[key] = (now, candidates)
            self._cache.move_to_end(key)
            self._prune_locked(now)

    def _schedule(self, request: Dict[str, Any]) -> None:
        queued = dict(request)
        queued["candidates"] = [dict(candidate) for candidate in request["candidates"]]
        queued["inference_trigger"] = "explicit"
        with self._condition:
            self._pending = queued
            self._pending_at = time.monotonic()
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                request = self._pending
                scheduled_at = self._pending_at
                delay = self.settle_seconds - (time.monotonic() - scheduled_at)
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    if self._pending is not request:
                        continue
                    if time.monotonic() - self._pending_at < self.settle_seconds:
                        continue
                if self._pending is not request:
                    continue
                self._pending = None
            if request is None:
                continue
            try:
                with self._inference_lock:
                    response = self._delegate.rank(request)
                self._store(request, response)
            except Exception:
                # Prefetch is opportunistic.  The explicit path remains the
                # correctness fallback and will surface/log backend failures.
                continue

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request = validate_request(request)
        trigger = request["inference_trigger"]
        if trigger in {"prefetch", "interactive"}:
            self._schedule(request)
            return self._original_order(request)

        cached = self._lookup(request)
        if cached is not None:
            return cached

        # If Space arrives before the typing debounce expires, discard that
        # queued copy and run the explicit request now instead of waiting 40 ms.
        key = self._key(request)
        with self._condition:
            if self._pending is not None and self._key(self._pending) == key:
                self._pending = None
                self._condition.notify_all()
        with self._inference_lock:
            response = self._delegate.rank(request)
        self._store(request, response)
        return response
