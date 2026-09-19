"""Production Dual Encoder with cached, one-shot joint segment decoding.

The original ONNX implementation lives in onnx_dual_encoder_base.py.  This
module changes only multi-segment ranking/prefetch; single-segment scoring and
public APIs remain inherited.  No additional model or training is required.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Sequence

from ranker.onnx_dual_encoder_base import (
    OnnxDualEncoderIMEReranker as _BaseRanker,
    content_signal_length,
    select_safe_local_context,
)

# A weak joint preference must never rewrite an otherwise plausible Mozc path.
_MIN_PATH_GAIN = 0.55
_MIN_PATH_MARGIN = 0.30
_CONFIDENT_PATH = 0.66  # Mozc's batch acceptance threshold is 0.65.


def _candidate_text(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("text", candidate.get("word", "")))


def _adjacent(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Never splice across a skipped (non-rerankable) Mozc segment."""
    try:
        before, after = str(previous["id"]), str(current["id"])
        return before.startswith("s") and after == f"s{int(before[1:]) + 1}"
    except (KeyError, ValueError):
        return False


def _alternative_prefix(
    previous: Dict[str, Any], current: Dict[str, Any], candidate: Dict[str, Any]
) -> str | None:
    """Replace the preceding segment's *original* surface, not the new top.

    Some Mozc candidate texts contain only a differing focus span; the common
    affix is already in preceding_text.  Retain any short trailing affix.
    If the baseline cannot be located, do not invent a segment boundary.
    """
    preceding = str(current.get("preceding_text", ""))
    previous_candidates = previous.get("candidates", [])
    if not previous_candidates or not _adjacent(previous, current):
        return None
    original = _candidate_text(previous_candidates[0])
    replacement = _candidate_text(candidate)
    if not original or not replacement:
        return None
    position = preceding.rfind(original)
    if position < 0 or len(preceding) - position - len(original) > 12:
        return None
    return preceding[:position] + replacement + preceding[position + len(original):]


def _context_plan(
    segments: Sequence[Dict[str, Any]], max_chars: int
) -> tuple[list[list[str]], list[str]]:
    """Prepare ALL unique context queries before a single encoder batch.

    Row i contains the query for each candidate of segment i-1.  Every query
    is keyed by its *exact safe text*, sharing both within-batch duplicates
    and embeddings prefetched while the user was typing.
    """
    rows: list[list[str]] = []
    unique: dict[str, None] = {}
    for i, segment in enumerate(segments):
        baseline = str(segment.get("preceding_text", ""))
        previous = segments[i - 1] if i else None
        alternatives = previous.get("candidates", []) if previous else [None]
        if not alternatives:
            alternatives = [None]
        row = []
        for candidate in alternatives:
            replacement = (
                _alternative_prefix(previous, segment, candidate)
                if previous is not None and candidate is not None else None
            )
            safe = select_safe_local_context(
                baseline if replacement is None else replacement,
                max_chars=max_chars,
            )
            row.append(safe)
            # Keep the original '文脈' sentinel for the zero-context cache.
            unique[safe if content_signal_length(safe) >= 2 else "文脈"] = None
        rows.append(row)
    return rows, list(unique)


def _softmax_confidence(values: Sequence[float]) -> float:
    if not values:
        return 0.5
    scaled = [float(value) * 4.0 for value in values]
    maximum = max(scaled)
    exps = [math.exp(value - maximum) for value in scaled]
    return max(exps) / sum(exps)


class _NonCacheableBatchResponse(dict):
    """Marker for a fallback that must be retried after prefetch completes."""

    cacheable = False


class OnnxDualEncoderIMEReranker(_BaseRanker):
    """Jointly select existing Mozc candidates without re-encoding contexts.

    This is a first-order (adjacent-segment) lattice, not unrestricted Cartesian
    full-sentence encoding.  Contexts beyond the immediate previous segment
    retain Mozc's initial surfaces.  This bounded approximation is important
    for interactive latency and avoids claiming full global semantics.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Coalesce concurrent prefetch/Space requests for the same context.
        self._joint_context_lock = threading.RLock()

    def _joint_context_vectors(self, queries: Sequence[str]):
        with self._joint_context_lock:
            return self._get_context_vectors(queries)

    @staticmethod
    def _baseline_batch_response(request: Dict[str, Any]) -> Dict[str, Any]:
        """Return Mozc order without starting any model work on Space."""
        output = []
        for index, segment in enumerate(request.get("segments", [])):
            candidates = segment.get("candidates", [])
            output.append({
                "id": str(segment.get("id", f"s{index}")),
                "winner_id": str(candidates[0]["id"]) if candidates else "",
                "confidence": 0.5,
            })
        return _NonCacheableBatchResponse({
            "request_id": request.get("request_id", "batch"),
            "segments": output,
        })

    @staticmethod
    def _prefetch_response(request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "request_id": request.get("request_id", "prefetch"),
            "segments": [
                {
                    "id": str(segment["id"]),
                    "winner_id": str(segment["candidates"][0]["id"]),
                    "confidence": 0.0,
                }
                for segment in request.get("segments", [])
            ],
        }

    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = request.get("request_id", "batch")
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request_id, "segments": []}

        rows, queries = _context_plan(segments, self.context_chars)
        if not any(
            content_signal_length(prefix) >= 2
            for row in rows
            for prefix in row
        ):
            return self._baseline_batch_response(request)
        words = [
            _candidate_text(candidate)
            for segment in segments
            for candidate in segment.get("candidates", [])
        ]
        # Space/Enter is read-only: never encode a context or candidate on the
        # pipe request thread.  If either side is still being prefetched, wait
        # briefly for the already-running workers.  A cache miss after that
        # wait is a safe Mozc-order fallback; it must not start new inference.
        with self._joint_context_lock:
            vectors = self._get_cached_context_vectors(queries)
        cache_ready = self._wait_for_prefetch_cache(queries, words)
        if not cache_ready:
            return self._baseline_batch_response(request)
        if vectors is None:
            with self._joint_context_lock:
                vectors = self._get_cached_context_vectors(queries)
        if vectors is None:
            return self._baseline_batch_response(request)
        by_query = dict(zip(queries, vectors))

        # scores[i][previous_candidate][current_candidate].  The first segment
        # has one dummy previous state.  No model inference in these loops.
        scores: list[list[list[float]]] = []
        for i, segment in enumerate(segments):
            candidates = segment.get("candidates", [])
            if not candidates:
                scores.append([[0.0]])
                continue
            alternatives: list[list[float]] = []
            for prefix in rows[i]:
                if content_signal_length(prefix) < 2:
                    # Preserve Mozc ordering when its context is too weak.
                    alternatives.append([-float(j) for j in range(len(candidates))])
                    continue
                scored = self._score_segment_candidates(
                    prefix, str(segment.get("following_text", "")),
                    str(segment.get("read", "")), candidates, by_query[prefix],
                    allow_encode=False,
                )
                if scored is None:
                    alternatives.append([-float(j) for j in range(len(candidates))])
                    continue
                by_id = {str(item["id"]): float(item["final_score"])
                         for item in scored}
                alternatives.append([
                    by_id[str(candidate["id"])] for candidate in candidates
                ])
            scores.append(alternatives)

        # Viterbi: retain two distinct best paths at every state, so a path
        # margin is measured against the real runner-up, not against Mozc only.
        previous_paths: list[list[tuple[float, tuple[int, ...]]]] = []
        for i, segment in enumerate(segments):
            candidates = segment.get("candidates", [])
            count = len(candidates) or 1
            current_paths: list[list[tuple[float, tuple[int, ...]]]] = []
            for j in range(count):
                if i == 0:
                    choices = [(scores[0][0][j], (j,))]
                else:
                    choices = [
                        (old_score + scores[i][min(prev_j, len(scores[i]) - 1)][j],
                         old_path + (j,))
                        for prev_j, prior in enumerate(previous_paths)
                        for old_score, old_path in prior
                    ]
                    choices.sort(key=lambda item: item[0], reverse=True)
                    choices = choices[:2]
                current_paths.append(choices)
            previous_paths = current_paths
        ranked_paths = sorted(
            (path for choices in previous_paths for path in choices),
            key=lambda item: item[0], reverse=True,
        )
        best_score, best_path = ranked_paths[0]
        second_score = ranked_paths[1][0] if len(ranked_paths) > 1 else -math.inf
        baseline_score = sum(
            scores[i][0][0] for i in range(len(segments))
        )
        joint_safe = (
            len(segments) > 1
            and best_score - baseline_score >= _MIN_PATH_GAIN
            and best_score - second_score >= _MIN_PATH_MARGIN
        )

        output = []
        for i, segment in enumerate(segments):
            candidates = segment.get("candidates", [])
            if not candidates:
                output.append({"id": str(segment.get("id", f"s{i}")),
                               "winner_id": "", "confidence": 0.5})
                continue
            previous = best_path[i - 1] if i and joint_safe else 0
            local = scores[i][min(previous, len(scores[i]) - 1)]
            if joint_safe:
                winner = best_path[i]
                # Commit all segments as one coherent choice: the C++ client
                # applies only confidence>=0.65, otherwise a partial path
                # would be inconsistent and trigger redundant model calls.
                confidence = _CONFIDENT_PATH
            else:
                winner = max(range(len(candidates)), key=lambda j: local[j])
                confidence = _softmax_confidence(local)
            output.append({
                "id": str(segment.get("id", f"s{i}")),
                "winner_id": str(candidates[winner]["id"]),
                "confidence": round(confidence, 4),
            })
        return {"request_id": request_id, "segments": output}

    def prefetch_context_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Warm the joint context side only, once per context generation."""
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request.get("request_id", "prefetch"),
                    "segments": []}
        _, queries = _context_plan(segments, self.context_chars)
        signature = tuple(queries)
        cache_lock = getattr(self, "_cache_lock", None)
        if cache_lock is None:
            if signature != getattr(self, "_context_prefetch_signature", ()):
                getattr(self, "cache", {}).clear()
                self._context_prefetch_signature = signature
        else:
            with cache_lock:
                if signature != self._context_prefetch_signature:
                    self.context_cache.clear()
                    self.context_cache_signature = ()
                    self._context_prefetch_signature = signature
        self._joint_context_vectors(queries)
        return self._prefetch_response(request)

    def prefetch_candidate_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Warm candidate embeddings only; no context encoder call here."""
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request.get("request_id", "prefetch"),
                    "segments": []}
        self.preload_candidates([
            _candidate_text(candidate)
            for segment in segments
            for candidate in segment.get("candidates", [])
            if _candidate_text(candidate)
        ])
        return self._prefetch_response(request)

    def prefetch_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible combined prefetch for older clients."""
        self.prefetch_context_batch(request)
        return self.prefetch_candidate_batch(request)
