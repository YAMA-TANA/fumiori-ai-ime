import time
from ranker.onnx_dual_encoder_ranker import OnnxDualEncoderIMEReranker

ranker = OnnxDualEncoderIMEReranker()
req = {
    'request_id': 'test-1',
    'preceding_text': '役員が稟議書を',
    'read': 'けっさい',
    'candidates': [
        {'id': 'c1', 'text': '決済', 'rank': 1},
        {'id': 'c2', 'text': '決裁', 'rank': 2},
        {'id': 'c3', 'text': '血清', 'rank': 3},
    ]
}
# warmup
ranker.rank(req)

# benchmark 10 iterations
latencies = []
for _ in range(10):
    res = ranker.rank(req)
    latencies.append(res['latency_ms'])

print('Winner:', res['winner_text'])
print('Average Latency ms:', sum(latencies) / len(latencies))
print('Min Latency ms:', min(latencies))
print('Breakdown:', res['breakdown'])
assert res['winner_text'] == '決裁', f"Expected 決裁, got {res['winner_text']}"
print('ALL TESTS PASSED!')
