import json
from ranker.ranker import process_line
from ranker.onnx_dual_encoder_ranker import OnnxDualEncoderIMEReranker


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
