import json
from pathlib import Path

import pytest

from ranker.ranker import process_line
from ranker.onnx_dual_encoder_ranker import OnnxDualEncoderIMEReranker


def _dual_encoder_model_available() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    return any(
        (repo_root / relative).is_file()
        for relative in (
            "build/onnx-model-70m-dual-encoder/dual-encoder-70m-int8.onnx",
            "build/onnx-model-70m-dual-encoder/dual-encoder-70m-fp16.onnx",
            "build/onnx-model-70m-dual-encoder/dual-encoder-70m-fp32.onnx",
            "models/onnx/dual-encoder-70m-int8.onnx",
            "models/onnx/dual-encoder-70m-fp16.onnx",
        )
    )


@pytest.mark.skipif(
    not _dual_encoder_model_available(),
    reason="Dual-Encoder ONNX integration test requires the optional model bundle",
)
def test_process_line_dual_encoder():
    ranker = OnnxDualEncoderIMEReranker()
    req = {
        'request_id': 'mozc-test-1',
        'inference_trigger': 'explicit',
        'preceding_text': '役員が稟議書を',
        'read': 'けっさい',
        'candidates': [
            {'id': 'c1', 'text': '決済', 'rank': 1},
            {'id': 'c2', 'text': '決裁', 'rank': 2},
            {'id': 'c3', 'text': '血清', 'rank': 3},
        ]
    }
    line = (json.dumps(req) + '\n').encode('utf-8')
    resp_bytes = process_line(line, ranker)
    resp = json.loads(resp_bytes.decode('utf-8'))
    assert resp['request_id'] == 'mozc-test-1'
    assert resp['candidates'][0]['id'] == 'c2'
    assert resp['candidates'][0]['rank'] == 1


def test_process_line_prefetch_does_not_call_candidate_ranker():
    class PrefetchOnly:
        def __init__(self):
            self.calls = 0

        def prefetch_batch(self, request):
            self.calls += 1
            assert request['segments'][0]['candidates'][0]['text'] == '測る'
            return {
                'request_id': request['request_id'],
                'segments': [{
                    'id': request['segments'][0]['id'],
                    'winner_id': request['segments'][0]['candidates'][0]['id'],
                    'confidence': 0.0,
                }],
            }

    ranker = PrefetchOnly()
    req = {
        'request_id': 'mozc-prefetch-1',
        'inference_trigger': 'prefetch',
        'segments': [{
            'id': 's0',
            'preceding_text': '月と地球の距離を',
            'following_text': '',
            'read': 'はかる',
            'candidates': [{'id': 'c0', 'text': '測る', 'rank': 1}],
        }],
    }
    response = process_line(
        (json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8'), ranker
    )
    assert ranker.calls == 1
    assert json.loads(response.decode('utf-8'))['segments'][0]['confidence'] == 0.0


def test_process_line_routes_split_prefetch_to_separate_workers():
    class SplitPrefetch:
        def __init__(self):
            self.context = 0
            self.candidates = 0

        def prefetch_context_batch_async(self, request):
            self.context += 1

        def prefetch_candidate_batch_async(self, request):
            self.candidates += 1

    ranker = SplitPrefetch()
    base = {
        'request_id': 'mozc-split-1',
        'segments': [{
            'id': 's0',
            'preceding_text': '月と地球の距離を',
            'following_text': '',
            'read': 'はかる',
            'candidates': [{'id': 'c0', 'text': '測る', 'rank': 1}],
        }],
    }
    for trigger in ('context_prefetch', 'candidate_prefetch'):
        request = dict(base, inference_trigger=trigger)
        response = process_line(
            (json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8'),
            ranker,
        )
        assert json.loads(response.decode('utf-8'))['segments'][0]['confidence'] == 0.0
    assert ranker.context == 1
    assert ranker.candidates == 1
