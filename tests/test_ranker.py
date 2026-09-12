from __future__ import annotations

import ctypes
import json
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from client.fallback import original_order, safe_rank
from ranker.protocol import (
    ProtocolError,
    loads_strict,
    validate_batch_request,
    validate_batch_response,
    validate_request,
    validate_response,
)
from ranker.lexicon import contextual_candidate_bonus
from ranker.ranker import (InteractiveBurstGuard, RuleBasedRanker,
                           ResponseCache,
                           _low_integrity_pipe_security,
                           normalize_windows_pipe_name, process_line,
                           summarize_reorder)
from ranker.qwen_ranker import QwenReranker
from client.windows_pipe import rank_once as windows_rank_once


ROOT = Path(__file__).resolve().parents[1]
FULL_MODEL_TESTS = (
    os.environ.get("YAMATANA_FULL_MODEL_TESTS") == "1"
    and (ROOT / "models" / "ruri-v3-reranker-310m-ime-tuned" / "model.safetensors").is_file()
)
HAS_TORCH = importlib.util.find_spec("torch") is not None
HEADLESS_TESTS = os.environ.get("YAMATANA_HEADLESS_TESTS") == "1"


def request(context: str):
    return {
        "request_id": "test-1",
        "preceding_text": context,
        "read": "はな",
        "candidates": [
            {"id": "c1", "text": "はな", "rank": 1},
            {"id": "c2", "text": "花", "rank": 2},
            {"id": "c3", "text": "鼻", "rank": 3},
        ],
    }


class RankerTests(unittest.TestCase):
    def test_interactive_burst_guard_defers_ai_until_request_settles(self):
        class FakeRanker:
            def __init__(self):
                self.calls = 0

            def rank(self, req):
                self.calls += 1
                candidates = list(reversed(req["candidates"]))
                return {
                    "request_id": req["request_id"],
                    "candidates": [
                        {"id": item["id"], "score": 10.0 - rank, "rank": rank}
                        for rank, item in enumerate(candidates, start=1)
                    ],
                }

        delegate = FakeRanker()
        guard = InteractiveBurstGuard(delegate, settle_seconds=0.01)
        req = request("庭には美しい")
        first = guard.rank(req)
        self.assertEqual([item["id"] for item in first["candidates"]], ["c1", "c2", "c3"])
        self.assertEqual(delegate.calls, 0)

        time.sleep(0.02)
        req["request_id"] = "test-2"
        settled = guard.rank(req)
        self.assertEqual([item["id"] for item in settled["candidates"]], ["c3", "c2", "c1"])
        self.assertEqual(delegate.calls, 1)

        req["request_id"] = "test-3"
        cached = guard.rank(req)
        self.assertEqual(cached["request_id"], "test-3")
        self.assertEqual(delegate.calls, 1)

    def test_interactive_burst_guard_runs_explicit_conversion_immediately(self):
        class FakeRanker:
            def __init__(self):
                self.calls = 0

            def rank(self, req):
                self.calls += 1
                candidates = list(reversed(req["candidates"]))
                return {
                    "request_id": req["request_id"],
                    "candidates": [
                        {"id": item["id"], "score": 10.0 - rank, "rank": rank}
                        for rank, item in enumerate(candidates, start=1)
                    ],
                }

        delegate = FakeRanker()
        guard = InteractiveBurstGuard(delegate, settle_seconds=0.5)
        req = request("庭には美しい")
        req["inference_trigger"] = "explicit"
        result = guard.rank(req)
        self.assertEqual([item["id"] for item in result["candidates"]], ["c3", "c2", "c1"])
        self.assertEqual(delegate.calls, 1)

    def test_protocol_normalizes_missing_trigger_as_interactive(self):
        req = validate_request(request("庭には美しい"))
        self.assertEqual(req["inference_trigger"], "interactive")

        explicit = request("庭には美しい")
        explicit["inference_trigger"] = "explicit"
        self.assertEqual(validate_request(explicit)["inference_trigger"], "explicit")

    def test_context_free_request_preserves_mozc_order_without_model_call(self):
        class UnexpectedRanker:
            def rank(self, _request):
                raise AssertionError("context-free request reached the model")

        req = request("")
        output = process_line(
            (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"),
            UnexpectedRanker(),
        )
        self.assertIsNotNone(output)
        response = loads_strict(output.decode("utf-8"))
        validate_response(response, validate_request(req))
        self.assertEqual(
            [candidate["id"] for candidate in response["candidates"]],
            ["c1", "c2", "c3"],
        )

    def test_explicit_suffix_only_request_reaches_model(self):
        class RecordingRanker:
            def __init__(self):
                self.calls = 0

            def rank(self, req):
                self.calls += 1
                reversed_candidates = list(reversed(req["candidates"]))
                return {
                    "request_id": req["request_id"],
                    "candidates": [
                        {"id": item["id"], "score": 10.0 - rank, "rank": rank}
                        for rank, item in enumerate(reversed_candidates, start=1)
                    ],
                }

        ranker = RecordingRanker()
        req = request("")
        req["following_text"] = "少年"
        req["inference_trigger"] = "explicit"
        output = process_line(
            (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"), ranker
        )
        self.assertIsNotNone(output)
        response = loads_strict(output.decode("utf-8"))
        self.assertEqual(ranker.calls, 1)
        self.assertEqual(
            [candidate["id"] for candidate in response["candidates"]],
            ["c3", "c2", "c1"],
        )

    def test_batch_protocol_returns_only_winner_and_confidence(self):
        req = {
            "request_id": "batch-protocol",
            "inference_trigger": "explicit",
            "segments": [{
                "id": "s0",
                "preceding_text": "庭には美しい",
                "following_text": "が咲く",
                "read": "はな",
                "candidates": [
                    {"id": "c0", "text": "花", "rank": 1},
                    {"id": "c1", "text": "鼻", "rank": 2},
                ],
            }],
        }
        normalized = validate_batch_request(req)
        response = {
            "request_id": "batch-protocol",
            "segments": [{"id": "s0", "winner_id": "c1", "confidence": 0.81}],
        }
        self.assertEqual(validate_batch_response(response, normalized), response)
        with self.assertRaises(ProtocolError):
            validate_batch_response({
                "request_id": "batch-protocol",
                "segments": [{
                    "id": "s0", "winner_id": "c1", "confidence": 0.81,
                    "score": 1.0,
                }],
            }, normalized)

    def test_response_cache_ignores_request_id_and_expires(self):
        class CountingRanker:
            def __init__(self):
                self.calls = 0

            def rank(self, req):
                self.calls += 1
                return {
                    "request_id": req["request_id"],
                    "candidates": [
                        {"id": item["id"], "score": float(-item["rank"]),
                         "rank": item["rank"]}
                        for item in req["candidates"]
                    ],
                }

        delegate = CountingRanker()
        cache = ResponseCache(delegate, ttl_seconds=0.02)
        first = request("庭には美しい")
        first["inference_trigger"] = "explicit"
        cached = dict(first)
        cached["request_id"] = "cache-2"
        cache.rank(first)
        result = cache.rank(cached)
        self.assertEqual(delegate.calls, 1)
        self.assertEqual(result["request_id"], "cache-2")
        time.sleep(0.03)
        cache.rank({**cached, "request_id": "cache-3"})
        self.assertEqual(delegate.calls, 2)

    def test_onnx_batch_flattens_all_segment_candidates_into_one_forward(self):
        import numpy as np
        from ranker.onnx_ranker import OnnxRuriReranker

        class Encoding:
            def __init__(self, index):
                self.ids = [index + 1]
                self.attention_mask = [1]

        class CapturingTokenizer:
            def __init__(self):
                self.pairs = []

            def encode_batch(self, pairs):
                self.pairs = list(pairs)
                return [Encoding(index) for index in range(len(self.pairs))]

        class CountingSession:
            def __init__(self):
                self.calls = 0
                self.batch_sizes = []

            def run(self, _outputs, inputs):
                self.calls += 1
                size = int(inputs["input_ids"].shape[0])
                self.batch_sizes.append(size)
                return [np.arange(size, dtype=np.float32).reshape(-1, 1)]

        tokenizer = CapturingTokenizer()
        session = CountingSession()
        ranker = object.__new__(OnnxRuriReranker)
        ranker.tokenizer = tokenizer
        ranker.session = session
        ranker.context_enabled = True
        ranker.context_chars = 128
        ranker.query_variant = "current"
        ranker.document_instruction = "一般的な日本語文書。"
        ranker.lexicon = None
        ranker.prior_w = 0.0
        ranker.safety_gate = False

        req = {
            "request_id": "two-segment-one-batch",
            "inference_trigger": "explicit",
            "segments": [
                {
                    "id": "s0", "preceding_text": "前", "following_text": "後",
                    "read": "はな",
                    "candidates": [
                        {"id": "c0", "text": "花", "rank": 1},
                        {"id": "c1", "text": "鼻", "rank": 2},
                        {"id": "c2", "text": "花", "rank": 3},
                    ],
                },
                {
                    "id": "s1", "preceding_text": "前花", "following_text": "終",
                    "read": "はし",
                    "candidates": [
                        {"id": "c0", "text": "橋", "rank": 1},
                        {"id": "c1", "text": "箸", "rank": 2},
                    ],
                },
            ],
        }
        response = ranker.rank_batch(req)

        self.assertEqual(session.calls, 1)
        self.assertEqual(session.batch_sizes, [4])
        self.assertEqual(len(tokenizer.pairs), 4)
        self.assertEqual(
            set(item["id"] for item in response["segments"]), {"s0", "s1"}
        )
        self.assertTrue(all(set(item) == {"id", "winner_id", "confidence"}
                            for item in response["segments"]))

    def test_onnx_scores_every_candidate_in_one_forward_batch(self):
        import numpy as np
        from ranker.onnx_ranker import OnnxRuriReranker

        class Encoding:
            def __init__(self, index):
                self.ids = [index + 1]
                self.attention_mask = [1]

        class CapturingTokenizer:
            def __init__(self):
                self.pairs = []

            def encode_batch(self, pairs):
                self.pairs = list(pairs)
                return [Encoding(index) for index in range(len(self.pairs))]

        class CountingSession:
            def __init__(self):
                self.calls = 0
                self.batch_sizes = []

            def run(self, _outputs, inputs):
                self.calls += 1
                size = int(inputs["input_ids"].shape[0])
                self.batch_sizes.append(size)
                return [np.arange(size, dtype=np.float32).reshape(-1, 1)]

        tokenizer = CapturingTokenizer()
        session = CountingSession()
        ranker = object.__new__(OnnxRuriReranker)
        ranker.tokenizer = tokenizer
        ranker.session = session
        ranker.context_enabled = True
        ranker.context_chars = 128
        ranker.document_instruction = "一般的な日本語文書。"
        ranker.lexicon = None
        ranker.prior_w = 0.0

        req = {
            "request_id": "all-candidates-one-batch",
            "preceding_text": "この語は",
            "following_text": "です",
            "read": "こうほ",
            "candidates": [
                {"id": f"c{index}", "text": text, "rank": index + 1}
                for index, text in enumerate(["候補", "公募", "候補", "後補", "甲保"])
            ],
        }
        response = ranker.rank(req)

        self.assertEqual(session.calls, 1)
        self.assertEqual(session.batch_sizes, [len(req["candidates"])])
        self.assertEqual(len(tokenizer.pairs), len(req["candidates"]))
        self.assertEqual(
            [document for _query, document in tokenizer.pairs],
            [f"この語は{item['text']}です" for item in req["candidates"]],
        )
        self.assertEqual(response["candidates"][0]["id"], "c4")
        explanation = ranker.last_explanation
        self.assertEqual(
            explanation["formula"],
            "evidence = model + style_bonus + context_bonus - lexical_penalty - reading_identity_penalty; final = evidence - rank_prior_penalty",
        )
        self.assertEqual(explanation["decision"]["selected_id"], "c4")
        self.assertEqual(explanation["candidates"][0]["text"], "甲保")
        self.assertEqual(explanation["candidates"][0]["style_bonus"], 0.0)

    def test_onnx_two_model_ensemble_uses_short_reading_calibration(self):
        import numpy as np
        from ranker.onnx_ranker import OnnxRuriReranker

        class Encoding:
            def __init__(self, index):
                self.ids = [index + 1]
                self.attention_mask = [1]

        class CapturingTokenizer:
            def encode_batch(self, pairs):
                return [Encoding(index) for index, _pair in enumerate(pairs)]

        class FixedSession:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
                self.calls = 0

            def run(self, _outputs, _inputs):
                self.calls += 1
                return [self.values]

        model_a = FixedSession([0.0, 2.0])
        model_b = FixedSession([2.0, 0.0])
        ranker = object.__new__(OnnxRuriReranker)
        ranker.tokenizer = CapturingTokenizer()
        ranker.sessions = [model_a, model_b]
        ranker.session = model_a
        ranker.ensemble_weights = [0.25, 0.75]
        ranker.model_paths = ["a.onnx", "b.onnx"]
        ranker.context_enabled = True
        ranker.context_chars = 128
        ranker.document_instruction = "一般的な日本語文書。"
        ranker.lexicon = None
        ranker.prior_w = 0.0
        ranker.safety_gate = False

        ranker.rank({
            "request_id": "ensemble-short-reading",
            "preceding_text": "文脈",
            "following_text": "",
            "read": "はな",
            "candidates": [
                {"id": "c0", "text": "花", "rank": 1},
                {"id": "c1", "text": "鼻", "rank": 2},
            ],
        })

        self.assertEqual(model_a.calls, 1)
        self.assertEqual(model_b.calls, 1)
        self.assertEqual(ranker.last_explanation["parameters"]["ensemble_weights"], [0.5, 0.5])
        self.assertEqual(
            ranker.last_explanation["formula"],
            "evidence = weighted_model_ensemble + style_bonus + context_bonus - lexical_penalty - reading_identity_penalty; final = evidence - rank_prior_penalty",
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows security descriptor")
    def test_pipe_security_descriptor_allows_low_integrity_owner(self):
        attributes, descriptor = _low_integrity_pipe_security()
        try:
            self.assertTrue(attributes.security_descriptor)
            self.assertEqual(attributes.security_descriptor, descriptor.value)
            self.assertTrue(ctypes.windll.advapi32.IsValidSecurityDescriptor(descriptor))
        finally:
            ctypes.windll.kernel32.LocalFree(descriptor)

    def test_windows_pipe_name_is_canonical(self):
        expected = r"\\.\pipe\ai_ime_ranker"
        self.assertEqual(normalize_windows_pipe_name("ai_ime_ranker"), expected)
        self.assertEqual(normalize_windows_pipe_name(expected), expected)
        self.assertEqual(
            normalize_windows_pipe_name(r"\\.\pipe\\.\pipe\ai_ime_ranker"),
            expected,
        )

    def test_reorder_summary_does_not_store_candidate_text(self):
        req = validate_request(request(""))
        response = {
            "request_id": req["request_id"],
            "candidates": [
                {"id": "c3", "score": 3.0, "rank": 1},
                {"id": "c1", "score": 2.0, "rank": 2},
                {"id": "c2", "score": 1.0, "rank": 3},
            ],
        }
        self.assertEqual(summarize_reorder(req, response), (True, 3))

    def test_acceptance_elephant_nose(self):
        result = RuleBasedRanker().rank(request("象の長い"))
        self.assertEqual(result["candidates"][0]["id"], "c3")
        self.assertNotIn("text", result["candidates"][0])

    def test_acceptance_garden_flower(self):
        result = RuleBasedRanker().rank(request("庭に咲いた美しい"))
        self.assertEqual(result["candidates"][0]["id"], "c2")

    def test_contextual_bonus_disambiguates_elephant_and_garden_sentences(self):
        self.assertGreater(contextual_candidate_bonus("象は が長い", "鼻"), 90.0)
        self.assertGreater(contextual_candidate_bonus("庭には美しい がある", "花"), 90.0)
        self.assertLess(contextual_candidate_bonus("庭には美しい がある", "鼻"), 10.0)
        self.assertGreater(contextual_candidate_bonus(" 少年", "非行"), 90.0)
        self.assertEqual(contextual_candidate_bonus(" 少年", "飛行"), 0.0)
        self.assertEqual(contextual_candidate_bonus("少年 ", "非行"), 0.0)

    def test_acceptance_machine_calculation(self):
        req = {
            "request_id": "test-3",
            "preceding_text": "電子工学の分野において、計算",
            "read": "きかい",
            "candidates": [
                {"id": "c1", "text": "機会", "rank": 1},
                {"id": "c2", "text": "機械", "rank": 2},
                {"id": "c3", "text": "器械", "rank": 3},
            ],
        }
        result = RuleBasedRanker().rank(req)
        self.assertEqual(result["candidates"][0]["id"], "c2")

    @unittest.skipUnless(FULL_MODEL_TESTS, "requires the full PyTorch integration model")
    def test_ruri_reranker_homophone_disambiguation(self):
        from ranker.ruri_ranker import RuriReranker
        ranker = RuriReranker()
        req = {
            "request_id": "ruri-1",
            "preceding_text": "医師の治療で長年の病気を",
            "following_text": "ことができた。",
            "read": "なおす",
            "candidates": [
                {"id": "c1", "text": "直す", "rank": 1},
                {"id": "c2", "text": "治す", "rank": 2},
            ],
        }
        res = ranker.rank(req)
        self.assertEqual(res["candidates"][0]["id"], "c2")

    @unittest.skipUnless(FULL_MODEL_TESTS, "requires the full PyTorch integration model")
    def test_ruri_reranker_lexical_grounding_shukanshi(self):
        from ranker.ruri_ranker import RuriReranker
        ranker = RuriReranker()
        req = {
            "request_id": "ruri-2",
            "preceding_text": "出張のたびに買っていたので、いつの間にか",
            "following_text": "を読む習慣が付いた。",
            "read": "しゅうかんし",
            "candidates": [
                {"id": "c1", "text": "週間誌", "rank": 1},
                {"id": "c2", "text": "週刊誌", "rank": 2},
            ],
        }
        res = ranker.rank(req)
        self.assertEqual(res["candidates"][0]["id"], "c2")

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_ruri_preserves_duplicate_surface_form_ids(self):
        import torch
        from ranker.ruri_ranker import RuriReranker

        class FakeTokenizer:
            def __init__(self):
                self.batch_size = 0

            def __call__(self, queries, documents, **_kwargs):
                self.batch_size = len(queries)
                return {"input_ids": torch.arange(len(queries)).reshape(-1, 1)}

        class FakeModel:
            def __call__(self, **inputs):
                return type("Output", (), {"logits": inputs["input_ids"].float()})()

        tokenizer = FakeTokenizer()
        ranker = object.__new__(RuriReranker)
        ranker.tokenizer = tokenizer
        ranker.model = FakeModel()
        ranker.device = "cpu"
        ranker.lexicon = None
        ranker.prior_w = 0.1

        req = {
            "request_id": "duplicate-ids",
            "preceding_text": "",
            "read": "はな",
            "candidates": [
                {"id": "c0", "text": "花", "rank": 1},
                {"id": "c1", "text": "花", "rank": 2},
                {"id": "c2", "text": "鼻", "rank": 3},
            ],
        }
        response = ranker.rank(req)
        validated = validate_response(response, validate_request(req))
        self.assertEqual(tokenizer.batch_size, 2)
        self.assertEqual({item["id"] for item in validated["candidates"]},
                         {"c0", "c1", "c2"})

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_ruri_passes_domain_and_custom_instruction_to_model(self):
        import torch
        from ranker.ruri_ranker import RuriReranker

        class CapturingTokenizer:
            def __init__(self):
                self.queries = []
                self.documents = []

            def __call__(self, queries, documents, **_kwargs):
                self.queries = list(queries)
                self.documents = list(documents)
                return {"input_ids": torch.arange(len(queries)).reshape(-1, 1)}

        class FakeModel:
            def __call__(self, **inputs):
                return type("Output", (), {"logits": inputs["input_ids"].float()})()

        tokenizer = CapturingTokenizer()
        ranker = object.__new__(RuriReranker)
        ranker.tokenizer = tokenizer
        ranker.model = FakeModel()
        ranker.device = "cpu"
        ranker.lexicon = None
        ranker.prior_w = 0.1
        ranker.context_enabled = True
        ranker.context_chars = 8
        ranker.document_instruction = "医学文書。循環器内科の正式表記を優先する。"

        req = request("これは切り捨てる長い文脈の末尾")
        ranker.rank(req)
        self.assertTrue(tokenizer.queries)
        self.assertIn("循環器内科", tokenizer.queries[0])
        self.assertNotIn("これは切り捨てる", tokenizer.queries[0])
        self.assertIn("長い文脈の末尾"[-8:], tokenizer.queries[0])

    @unittest.skipUnless(not HEADLESS_TESTS, "native loading UI is not started on headless CI")
    def test_loading_indicator_lifecycle(self):
        from ranker.loading_ui import LoadingIndicator
        indicator = LoadingIndicator(text="AI変換中…", enabled=True)
        indicator.start()
        with indicator.active():
            pass
        indicator.stop()

    def test_protocol_rejects_candidate_generation(self):
        req = validate_request(request(""))
        bad = {
            "request_id": req["request_id"],
            "candidates": [{"id": "new", "text": "生成文字列", "score": 1, "rank": 1}],
        }
        with self.assertRaises(ProtocolError):
            validate_response(bad, req)

    def test_protocol_rejects_unknown_id_and_nan(self):
        req = validate_request(request(""))
        bad = {"request_id": req["request_id"], "candidates": [
            {"id": "unknown", "score": 1, "rank": 1},
            {"id": "c2", "score": float("nan"), "rank": 2},
            {"id": "c3", "score": 0, "rank": 3},
        ]}
        with self.assertRaises(ProtocolError):
            validate_response(bad, req)

    def test_acceptance_following_text_picks_flower(self):
        req = {
            "request_id": "test-ft",
            "preceding_text": "庭には美しい",
            "following_text": "が咲く",
            "read": "はな",
            "candidates": [
                {"id": "c1", "text": "はな", "rank": 1},
                {"id": "c2", "text": "花", "rank": 2},
                {"id": "c3", "text": "鼻", "rank": 3},
            ],
        }
        result = RuleBasedRanker().rank(req)
        self.assertEqual(result["candidates"][0]["id"], "c2")

    def test_acceptance_following_text_picks_nose(self):
        req = {
            "request_id": "test-ft",
            "preceding_text": "彼は",
            "following_text": "が大きい",
            "read": "はな",
            "candidates": [
                {"id": "c1", "text": "はな", "rank": 1},
                {"id": "c2", "text": "花", "rank": 2},
                {"id": "c3", "text": "鼻", "rank": 3},
            ],
        }
        result = RuleBasedRanker().rank(req)
        self.assertEqual(result["candidates"][0]["id"], "c3")

    def test_protocol_rejects_duplicate_fields_and_non_contiguous_input_rank(self):
        with self.assertRaises(ProtocolError):
            loads_strict('{"request_id":"x","request_id":"y"}')
        req = request("")
        req["candidates"][1]["rank"] = 3
        with self.assertRaises(ProtocolError):
            validate_request(req)

    def test_invalid_and_timeout_fallback_to_original(self):
        req = request("象の長い")
        fallback = safe_rank(req, lambda _req: {"bad": True}, timeout_ms=200)
        self.assertEqual([item["id"] for item in fallback["candidates"]], ["c1", "c2", "c3"])
        fallback = safe_rank(req, lambda _req: (_ for _ in ()).throw(KeyError("backend")), timeout_ms=200)
        self.assertEqual(fallback, original_order(req))
        started = time.perf_counter()
        fallback = safe_rank(req, lambda _req: (time.sleep(0.5) or RuleBasedRanker().rank(req)), timeout_ms=30)
        elapsed = time.perf_counter() - started
        self.assertEqual(fallback, original_order(req))
        self.assertLess(elapsed, 0.2)

    def test_stdio_e2e(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.Popen(
            [sys.executable, "-m", "ranker.ranker", "--stdio",
             "--backend", "rule"],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write((json.dumps(request("庭に咲いた美しい"), ensure_ascii=False) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        proc.terminate()
        proc.wait(timeout=2)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
        response = json.loads(line.decode())
        validate_response(response, validate_request(request("庭に咲いた美しい")))
        self.assertEqual(response["candidates"][0]["id"], "c2")

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_qwen_backend_uses_batched_forward_and_existing_ids_only(self):
        import torch

        class FakeTokenizer:
            def convert_tokens_to_ids(self, token):
                return {"no": 0, "yes": 1}[token]

            def encode(self, text, add_special_tokens=False):
                return [9] if "system" in text else [8]

            def __call__(self, texts, **kwargs):
                # A marker stands in for each supplied document; it is not a
                # candidate produced by the backend.
                markers = []
                for text in texts:
                    document = text.rsplit("<Document>: ", 1)[-1]
                    markers.append(2 if document == "はな" else
                                   3 if document == "花" else 4)
                return {"input_ids": [[marker] for marker in markers]}

            def pad(self, encoded, **kwargs):
                return {"input_ids": torch.tensor(encoded["input_ids"])}

        class FakeModel:
            def __init__(self):
                self.forward_calls = 0

            def eval(self):
                return self

            def __call__(self, **inputs):
                self.forward_calls += 1
                ids = inputs["input_ids"]
                logits = torch.zeros((ids.shape[0], ids.shape[1], 5))
                # The fake model scores each input row independently in one
                # forward call, with no text output API.
                logits[:, -1, 1] = ids[:, 1].float()
                return type("Output", (), {"logits": logits})()

        model = FakeModel()
        ranker = QwenReranker(
            tokenizer=FakeTokenizer(), model=model, torch_module=torch,
        )
        result = ranker.rank(request("庭に咲いた美しい"))
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(result["candidates"][0]["id"], "c3")
        self.assertTrue(all(set(item) == {"id", "score", "rank"}
                            for item in result["candidates"]))

    @unittest.skipUnless(sys.platform == "win32", "Windows Named Pipe smoke test")
    def test_windows_named_pipe_e2e(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pipe = r"\\.\pipe\ai_ime_ranker_test_%d" % os.getpid()
        proc = subprocess.Popen(
            [sys.executable, "-m", "ranker.ranker", "--pipe", pipe,
             "--backend", "rule"],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            deadline = time.time() + 3
            while True:
                try:
                    response = windows_rank_once(pipe, request("象の長い"), timeout_ms=100)
                    break
                except (OSError, TimeoutError):
                    if time.time() > deadline:
                        raise
                    time.sleep(0.02)
            self.assertEqual(response["candidates"][0]["id"], "c3")
        finally:
            proc.terminate()
            proc.wait(timeout=2)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
