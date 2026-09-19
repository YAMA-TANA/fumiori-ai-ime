"""Benchmark multi-segment conversion: Cross-Encoder vs Dual-Encoder All-Dot-Product."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

# Avoid sklearn import on Windows
_find_spec = importlib.util.find_spec
def _find_spec_without_sklearn(name: str, *args: Any, **kwargs: Any):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _find_spec(name, *args, **kwargs)
importlib.util.find_spec = _find_spec_without_sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker
from ranker.dual_encoder_ranker import DualEncoderIMEReranker

MULTI_SEGMENT_CASES = [
    {
        "label": "業務・3文節一括変換 (稟議決裁・資料確認)",
        "segments": [
            {
                "id": "s0",
                "preceding_text": "先週の経営会議において、",
                "read": "けっさいされた",
                "candidates": ["決裁された", "決済された", "けっさいされた"],
                "expected": "決裁された",
            },
            {
                "id": "s1",
                "preceding_text": "先週の経営会議において、決裁された",
                "read": "しりょうを",
                "candidates": ["資料を", "史料を", "飼料を"],
                "expected": "資料を",
            },
            {
                "id": "s2",
                "preceding_text": "先週の経営会議において、決裁された資料を",
                "read": "かくにんした",
                "candidates": ["確認した", "各員した", "かくにんした"],
                "expected": "確認した",
            },
        ],
    },
    {
        "label": "日常・3文節一括変換 (春の庭園・花鑑賞)",
        "segments": [
            {
                "id": "s0",
                "preceding_text": "春のポカポカと暖かい日差しの中で、",
                "read": "はなが",
                "candidates": ["花が", "鼻が", "華が"],
                "expected": "花が",
            },
            {
                "id": "s1",
                "preceding_text": "春のポカポカと暖かい日差しの中で、花が",
                "read": "きれいに",
                "candidates": ["綺麗に", "きれいに", "奇麗に"],
                "expected": "きれいに",
            },
            {
                "id": "s2",
                "preceding_text": "春のポカポカと暖かい日差しの中で、花がきれいに",
                "read": "さいた",
                "candidates": ["咲いた", "裂いた", "割いた"],
                "expected": "咲いた",
            },
        ],
    },
    {
        "label": "IT技術・4文節一括変換 (クラウド移行・システム試行)",
        "segments": [
            {
                "id": "s0",
                "preceding_text": "オンプレミスの旧サーバーから、",
                "read": "くらうどへ",
                "candidates": ["クラウドへ", "暗雲へ", "くらうどへ"],
                "expected": "クラウドへ",
            },
            {
                "id": "s1",
                "preceding_text": "オンプレミスの旧サーバーから、クラウドへ",
                "read": "いこうして",
                "candidates": ["移行して", "以降して", "意向して"],
                "expected": "移行して",
            },
            {
                "id": "s2",
                "preceding_text": "オンプレミスの旧サーバーから、クラウドへ移行して",
                "read": "しすてむを",
                "candidates": ["システムを", "しすてむを"],
                "expected": "システムを",
            },
            {
                "id": "s3",
                "preceding_text": "オンプレミスの旧サーバーから、クラウドへ移行してシステムを",
                "read": "しこうした",
                "candidates": ["試行した", "思考した", "施行した"],
                "expected": "試行した",
            },
        ],
    },
]


def run_multi_segment_benchmark():
    print("=" * 80)
    print("BENCHMARK: MULTI-SEGMENT CONVERSION (CROSS-ENCODER VS DUAL-ENCODER ALL-DOT-PRODUCT)")
    print("=" * 80)

    # 1. Initialize Cross-Encoder
    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    cross_ranker = OnnxRuriReranker(
        model_paths=[lora3_fp16, lora6_fp16],
        ensemble_weights=[0.25, 0.75],
        prior_w=0.0,
    )

    # 2. Initialize Dual-Encoder
    dual_model_dir = ROOT / "build" / "dual-encoder-70m"
    dual_ranker = DualEncoderIMEReranker(dual_model_dir)

    # Preload all candidates in benchmark
    all_words = set()
    for item in MULTI_SEGMENT_CASES:
        for seg in item["segments"]:
            all_words.update(seg["candidates"])
    dual_ranker.preload_candidates(list(all_words))
    print(f"Preloaded {len(all_words)} candidate vectors into resident cache.")

    # Warmup
    warmup_req = {
        "request_id": "warmup",
        "inference_trigger": "explicit",
        "segments": [
            {
                "id": "s0",
                "preceding_text": "テスト文脈",
                "following_text": "",
                "read": "てすと",
                "candidates": [{"id": "c1", "text": "テスト", "rank": 1}],
            }
        ],
    }
    cross_ranker.rank_batch(warmup_req)
    dual_ranker.rank_batch(warmup_req)

    print("\n" + "-" * 80)
    print("EVALUATING MULTI-SEGMENT CASES...")
    print("-" * 80)

    for case_idx, case in enumerate(MULTI_SEGMENT_CASES, start=1):
        label = case["label"]
        segments = case["segments"]
        total_candidates = sum(len(s["candidates"]) for s in segments)
        print(f"\n[{case_idx}] {label} ({len(segments)} segments, {total_candidates} total candidates)")

        cross_req = {
            "request_id": f"cross-{case_idx}",
            "inference_trigger": "explicit",
            "segments": [
                {
                    "id": seg["id"],
                    "preceding_text": seg["preceding_text"],
                    "following_text": "",
                    "read": seg["read"],
                    "candidates": [
                        {"id": f"{seg['id']}_c{c_idx}", "text": text, "rank": c_idx + 1}
                        for c_idx, text in enumerate(seg["candidates"])
                    ],
                }
                for seg in segments
            ],
        }

        # Run Cross-Encoder
        t0 = time.perf_counter()
        c_res = cross_ranker.rank_batch(cross_req)
        c_lat = (time.perf_counter() - t0) * 1000.0

        # Run Dual-Encoder
        dual_req = {
            "request_id": f"dual-{case_idx}",
            "inference_trigger": "explicit",
            "segments": [
                {
                    "id": seg["id"],
                    "preceding_text": seg["preceding_text"],
                    "following_text": "",
                    "read": seg["read"],
                    "candidates": [
                        {"id": f"{seg['id']}_c{c_idx}", "text": text, "rank": c_idx + 1}
                        for c_idx, text in enumerate(seg["candidates"])
                    ],
                }
                for seg in segments
            ],
        }
        t0 = time.perf_counter()
        d_res = dual_ranker.rank_batch(dual_req)
        d_lat = (time.perf_counter() - t0) * 1000.0

        print(f"  * Cross-Encoder Latency : {c_lat:6.2f} ms (Flattened huge Transformer batch)")
        print(f"  * Dual-Encoder Latency  : {d_lat:6.2f} ms (Context enc: {d_res['breakdown']['batch_context_encoding_ms']:.2f} ms, ALL-DOT: {d_res['breakdown']['all_dot_product_ms']:.4f} ms)")
        print(f"  * Speedup               : {c_lat / d_lat:.1f}x faster overall! (Dot-Product alone is {c_lat / max(0.001, d_res['breakdown']['all_dot_product_ms']):.0f}x faster)")

        # Inspect segment winners
        print("  * Results per segment:")
        for s_idx, seg in enumerate(segments):
            c_seg = c_res["segments"][s_idx]
            d_seg = d_res["segments"][s_idx]
            cand_map = {f"{seg['id']}_c{i}": text for i, text in enumerate(seg["candidates"])}
            c_winner = cand_map[c_seg["winner_id"]]
            d_winner = d_seg["winner_text"]
            expected = seg["expected"]

            c_ok = (c_winner == expected)
            d_ok = (d_winner == expected)

            print(f"    - Seg {s_idx+1} [{seg['read']}]: Expected: '{expected}'")
            print(f"        Cross-Encoder -> '{c_winner}' {'[OK]' if c_ok else '[NG]'}")
            print(f"        Dual-Encoder  -> '{d_winner}' {'[OK]' if d_ok else '[NG]'}")


if __name__ == "__main__":
    run_multi_segment_benchmark()
