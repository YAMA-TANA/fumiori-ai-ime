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

