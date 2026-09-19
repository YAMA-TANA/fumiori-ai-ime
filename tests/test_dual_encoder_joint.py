"""Dependency-free regression tests for the production ONNX joint wrapper."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


class FakeBase:
    context_chars = 36

    def __init__(self):
        self.cache = {}
        self.candidate_cache = set()
        self.encoding_batches = []
        self.context_calls = 0
        self.candidate_preloads = []
        self.wait_calls = []

    def _get_context_vectors(self, queries):
        self.context_calls += 1
        missing = list(dict.fromkeys(q for q in queries if q not in self.cache))
        if missing:
            self.encoding_batches.append(tuple(missing))
            for query in missing:
                self.cache[query] = np.array([len(query)], dtype=float)
        return np.stack([self.cache[q] for q in queries])

    def _get_cached_context_vectors(self, queries):
        if any(query not in self.cache for query in queries):
            return None
        return np.stack([self.cache[query] for query in queries])

    def _wait_for_prefetch_cache(self, queries, words, **kwargs):
        self.wait_calls.append((tuple(queries), tuple(words)))
        return (all(query in self.cache for query in queries)
                and all(word in self.candidate_cache for word in words if word))

    def _prefetch_active(self):
        return False

    def prefetch_batch_async(self, request):
        self.prefetch_batch(request)

    def preload_candidates(self, words):
        self.candidate_preloads.append(tuple(words))
        self.candidate_cache.update(word for word in words if word)

    def _score_segment_candidates(self, prefix, suffix, reading, candidates, vector,
                                  **kwargs):
        values = []
        for candidate in candidates:
            text = candidate['text']
            if reading == 'げんこうの':
                score = {'現行の': 1.2, '原稿の': 1.0}[text]
            elif reading == 'こうせい':
                if '原稿の' in prefix:
                    score = {'構成': 0.1, '校正': 3.0}[text]
                else:
                    score = {'構成': 1.1, '校正': 0.2}[text]
            elif reading == 'はな':
                score = {'花': 0.0, '鼻': 3.0}[text] if '顔' in prefix else 0.0
            elif reading == 'はかる':
                score = {'図る': 0.0, '測る': 3.0}[text] if '距離' in prefix else 0.0
            else:
                score = 0.0
            values.append({'id': candidate['id'], 'final_score': score})
        return sorted(values, key=lambda row: -row['final_score'])


def select_safe_local_context(text, max_chars=36):
    return text[-max_chars:]


def content_signal_length(text):
    return len(text.strip())


def load_wrapper(monkeypatch):
    ranker = types.ModuleType('ranker')
    ranker.__path__ = []
    base = types.ModuleType('ranker.onnx_dual_encoder_base')
    base.OnnxDualEncoderIMEReranker = FakeBase
    base.content_signal_length = content_signal_length
    base.select_safe_local_context = select_safe_local_context
    monkeypatch.setitem(sys.modules, 'ranker', ranker)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    path = Path(__file__).resolve().parents[1] / 'ranker' / 'onnx_dual_encoder_ranker.py'
    spec = importlib.util.spec_from_file_location('ranker.onnx_dual_encoder_ranker', path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def example():
    return {'request_id': 'example', 'segments': [
        {'id': 's1', 'preceding_text': '小説家は', 'following_text': '構成',
         'read': 'げんこうの', 'candidates': [
             {'id': 'c0', 'text': '現行の', 'rank': 1},
             {'id': 'c1', 'text': '原稿の', 'rank': 2}]},
        {'id': 's2', 'preceding_text': '小説家は現行の', 'following_text': '',
         'read': 'こうせい', 'candidates': [
             {'id': 'c0', 'text': '構成', 'rank': 1},
             {'id': 'c1', 'text': '校正', 'rank': 2}]},
    ]}


def test_joint_corrects_two_wrong_mozc_tops_in_one_context_batch(monkeypatch):
    mod = load_wrapper(monkeypatch)
    runner = mod.OnnxDualEncoderIMEReranker()
    request = example()
    runner.prefetch_batch(request)
    result = runner.rank_batch(request)
    assert [s['winner_id'] for s in result['segments']] == ['c1', 'c1']
    assert len(runner.encoding_batches) == 1
    assert set(runner.encoding_batches[0]) == {
        '小説家は', '小説家は現行の', '小説家は原稿の'
    }
    assert runner.context_calls == 1
    assert len(runner.candidate_preloads) == 1
    assert all(s['confidence'] >= .65 for s in result['segments'])


def test_prefetch_reuses_exact_context_embeddings_without_new_encoding(monkeypatch):
    mod = load_wrapper(monkeypatch)
    runner = mod.OnnxDualEncoderIMEReranker()
    req = example()
    prefetched = runner.prefetch_batch(req)
    assert len(runner.encoding_batches) == 1
    assert all(s['confidence'] == 0 for s in prefetched['segments'])
    ranked = runner.rank_batch(req)
    assert [s['winner_id'] for s in ranked['segments']] == ['c1', 'c1']
    assert len(runner.encoding_batches) == 1


def test_nonadjacent_segment_does_not_splice_across_omitted_segment(monkeypatch):
    mod = load_wrapper(monkeypatch)
    request = example()
    request['segments'][1]['id'] = 's3'
    rows, unique = mod._context_plan(request['segments'], 36)
    assert rows[1] == ['小説家は現行の', '小説家は現行の']
    assert '小説家は原稿の' not in unique


def test_focus_affix_is_preserved_when_previous_surface_is_replaced(monkeypatch):
    mod = load_wrapper(monkeypatch)
    request = example()
    request['segments'][0]['candidates'][0]['text'] = '現行'
    request['segments'][0]['candidates'][1]['text'] = '原稿'
    request['segments'][1]['preceding_text'] = '小説家は現行の原文'
    rows, _ = mod._context_plan(request['segments'], 36)
    assert rows[1] == ['小説家は現行の原文', '小説家は原稿の原文']


def test_no_context_preserves_mozc_top(monkeypatch):
    mod = load_wrapper(monkeypatch)
    runner = mod.OnnxDualEncoderIMEReranker()
    request = example()
    request['segments'][0]['preceding_text'] = ''
    request['segments'][1]['preceding_text'] = ''
    result = runner.rank_batch(request)
    assert [s['winner_id'] for s in result['segments']] == ['c0', 'c0']
    assert len(runner.encoding_batches) == 0


def test_explicit_conversion_completes_cache_misses(monkeypatch):
    mod = load_wrapper(monkeypatch)
    runner = mod.OnnxDualEncoderIMEReranker()
    result = runner.rank_batch(example())
    assert [s['winner_id'] for s in result['segments']] == ['c1', 'c1']
    assert len(runner.encoding_batches) == 1
    assert len(runner.candidate_preloads) == 1
    assert len(runner.wait_calls) == 1


def test_first_explicit_conversion_reranks_face_and_distance_examples(monkeypatch):
    mod = load_wrapper(monkeypatch)
    runner = mod.OnnxDualEncoderIMEReranker()
    for reading, prefix, original, expected in (
        ('はな', '彼の顔の', '花', '鼻'),
        ('はかる', '月と地球の距離を', '図る', '測る'),
    ):
        request = {'request_id': reading, 'segments': [{
            'id': 's0', 'preceding_text': prefix, 'following_text': '',
            'read': reading, 'candidates': [
                {'id': 'c0', 'text': original, 'rank': 1},
                {'id': 'c1', 'text': expected, 'rank': 2},
            ],
        }]}
        result = runner.rank_batch(request)
        assert result['segments'][0]['winner_id'] == 'c1'
        assert result['segments'][0]['confidence'] >= .65
