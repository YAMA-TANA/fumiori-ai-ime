from __future__ import annotations

import time
import unittest

from ranker.protocol import validate_request
from ranker.speculative import SpeculativeRanker


class FakeRanker:
    context_enabled = True
    context_chars = 64

    def __init__(self) -> None:
        self.calls = 0

    def rank(self, request):
        self.calls += 1
        candidates = list(reversed(request["candidates"]))
        return {
            "request_id": request["request_id"],
            "candidates": [
                {"id": item["id"], "score": float(10 - rank), "rank": rank}
                for rank, item in enumerate(candidates, start=1)
            ],
        }


def make_request(trigger: str, request_id: str = "r1"):
    return {
        "request_id": request_id,
        "inference_trigger": trigger,
        "preceding_text": "前文" * 40,
        "following_text": "後文",
        "read": "はな",
        "candidates": [
            {"id": "c0", "text": "花", "rank": 1},
            {"id": "c1", "text": "鼻", "rank": 2},
        ],
    }


class SpeculativeRankerTests(unittest.TestCase):
    def test_protocol_accepts_prefetch_trigger(self) -> None:
        normalized = validate_request(make_request("prefetch"))
        self.assertEqual(normalized["inference_trigger"], "prefetch")

    def test_prefetch_returns_immediately_then_explicit_hits_cache(self) -> None:
        backend = FakeRanker()
        ranker = SpeculativeRanker(backend, settle_seconds=0.0)

        started = time.perf_counter()
        immediate = ranker.rank(make_request("prefetch"))
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertEqual([x["id"] for x in immediate["candidates"]], ["c0", "c1"])

        deadline = time.monotonic() + 1.0
        while backend.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(backend.calls, 1)
        # Give the worker one scheduling slice to publish the completed result.
        time.sleep(0.01)

        explicit = make_request("explicit", request_id="r2")
        response = ranker.rank(explicit)
        self.assertEqual([x["id"] for x in response["candidates"]], ["c1", "c0"])
        self.assertEqual(backend.calls, 1)
        self.assertEqual(response["request_id"], "r2")

    def test_cache_key_uses_only_configured_local_context(self) -> None:
        backend = FakeRanker()
        ranker = SpeculativeRanker(backend, settle_seconds=0.0)
        first = validate_request(make_request("explicit", request_id="r1"))
        second = validate_request(make_request("explicit", request_id="r2"))
        # Change only text older than the last 64 characters.
        second["preceding_text"] = "別の古い文脈" * 20 + first["preceding_text"][-64:]
        self.assertEqual(ranker._key(first), ranker._key(second))


if __name__ == "__main__":
    unittest.main()
