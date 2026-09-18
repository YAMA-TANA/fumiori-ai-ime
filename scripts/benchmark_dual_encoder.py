"""Benchmark and compare Cross-Encoder (Current) vs Dual-Encoder (New)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

# Avoid sklearn import on Windows
_find_spec = importlib.util.find_spec
def _find_spec_without_sklearn(name: str, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _find_spec(name, *args, **kwargs)
importlib.util.find_spec = _find_spec_without_sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker
from ranker.dual_encoder_ranker import DualEncoderIMEReranker

BENCHMARK_CASES = [
    {
        "label": "業務・稟議決裁",
        "prefix": "役員が稟議書を",
        "reading": "けっさい",
        "candidates": ["決済", "決裁", "けっさい", "決裁者", "血清"],
        "mozc_top": "決済",
        "expected": "決裁",
    },
    {
        "label": "日常・身体の鼻",
        "prefix": "彼の顔の真ん中にある",
        "reading": "はな",
        "candidates": ["花", "鼻", "華", "端", "ハナ"],
        "mozc_top": "花",
        "expected": "鼻",
    },
    {
        "label": "IT技術・システム移行",
        "prefix": "オンプレミスの旧サーバーからクラウド環境へ",
        "reading": "いこう",
        "candidates": ["以降", "移行", "意向", "威光", "イコウ"],
        "mozc_top": "以降",
        "expected": "移行",
    },
    {
        "label": "法律・施行",
        "prefix": "改正された新制度が来週から",
        "reading": "しこう",
        "candidates": ["思考", "施行", "試行", "指向", "志向"],
        "mozc_top": "思考",
        "expected": "施行",
    },
    {
        "label": "医療・病気を治す",
        "prefix": "処方された抗生物質で病気を",
        "reading": "なおす",
        "candidates": ["直す", "治す", "猶す", "なおす"],
        "mozc_top": "直す",
        "expected": "治す",
    },
    {
        "label": "日常・天気の雨",
        "prefix": "外は暗くなり大粒の雨が",
        "reading": "ふる",
        "candidates": ["フル", "降る", "振る", "狂る", "ふる"],
        "mozc_top": "フル",
        "expected": "降る",
    },
    {
        "label": "業務・製品仕様",
        "prefix": "開発に着手する前に製品の",
        "reading": "しよう",
        "candidates": ["使用", "仕様", "試用", "私用", "しよう"],
        "mozc_top": "使用",
        "expected": "仕様",
    },
    {
        "label": "業務・原稿校正",
        "prefix": "公開前に編集者が記事を",
        "reading": "こうせい",
        "candidates": ["構成", "校正", "公正", "更生", "こうせい"],
        "mozc_top": "構成",
        "expected": "校正",
    },
    {
        "label": "業務・経費精算",
        "prefix": "出張から戻り交通費と経費を",
        "reading": "せいさん",
        "candidates": ["生産", "精算", "清算", "凄惨", "せいさん"],
        "mozc_top": "生産",
        "expected": "精算",
    },
    {
        "label": "製造・損害補償",
        "prefix": "事故による損害の",
        "reading": "ほしょう",
        "candidates": ["保証", "補償", "保障", "ほしょう"],
        "mozc_top": "保証",
        "expected": "補償",
    }
]


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: CROSS-ENCODER (CURRENT) VS DUAL-ENCODER (NEW)")
    print("=" * 80)

    # 1. Initialize Cross-Encoder
    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    print("\n[1] Initializing Cross-Encoder (Ruri-70M LoRA3+LoRA6 Ensemble)...")
    cross_ranker = OnnxRuriReranker(
        model_paths=[lora3_fp16, lora6_fp16],
        ensemble_weights=[0.25, 0.75],
        prior_w=0.0,
    )

    # 2. Initialize Dual-Encoder
    dual_model_dir = ROOT / "build" / "dual-encoder-70m"
    print("\n[2] Initializing Dual-Encoder (70M ModernBERT Dot-Product)...")
    dual_ranker = DualEncoderIMEReranker(dual_model_dir)

    # Pre-cache all candidates in memory (simulating IME resident vector store)
    all_words = set()
    for tc in BENCHMARK_CASES:
        all_words.update(tc["candidates"])
    dual_ranker.preload_candidates(list(all_words))
    print(f"Preloaded {len(all_words)} candidate vectors into resident memory cache.")

    # Warmup both
    for tc in BENCHMARK_CASES[:2]:
        req = {
            "request_id": "warmup",
            "preceding_text": tc["prefix"],
            "read": tc["reading"],
            "candidates": [{"id": f"c{j}", "text": cand, "rank": j+1} for j, cand in enumerate(tc["candidates"])]
        }
        cross_ranker.rank(req)
        dual_ranker.rank(req)

    print("\n" + "-" * 80)
    print(f"RUNNING EVALUATION ON {len(BENCHMARK_CASES)} PRACTICAL CASES...")
    print("-" * 80)

    cross_latencies = []
    cross_correct = 0
    dual_latencies = []
    dual_dot_latencies = []
    dual_correct = 0

    results = []

    for i, tc in enumerate(BENCHMARK_CASES, start=1):
        req = {
            "request_id": f"bench-{i}",
            "preceding_text": tc["prefix"],
            "read": tc["reading"],
            "candidates": [{"id": f"c{j}", "text": cand, "rank": j+1} for j, cand in enumerate(tc["candidates"])]
        }

        # Run Cross-Encoder
        t_c0 = time.perf_counter()
        c_res = cross_ranker.rank(req)
        c_lat = (time.perf_counter() - t_c0) * 1000.0
        cross_latencies.append(c_lat)
        
        cand_map = {c["id"]: c["text"] for c in req["candidates"]}
        c_winner = cand_map[c_res["candidates"][0]["id"]]
        c_ok = (c_winner == tc["expected"])
        cross_correct += int(c_ok)

        # Run Dual-Encoder
        d_res = dual_ranker.rank(req)
        dual_latencies.append(d_res["latency_ms"])
        dual_dot_latencies.append(d_res["breakdown"]["dot_product_ms"])
        d_winner = d_res["winner_text"]
        d_ok = (d_winner == tc["expected"])
        dual_correct += int(d_ok)

        results.append({
            "label": tc["label"],
            "prefix": tc["prefix"],
            "reading": tc["reading"],
            "mozc": tc["mozc_top"],
            "expected": tc["expected"],
            "cross_winner": c_winner,
            "cross_ok": c_ok,
            "cross_ms": c_lat,
            "dual_winner": d_winner,
            "dual_ok": d_ok,
            "dual_ms": d_res["latency_ms"],
            "dual_dot_ms": d_res["breakdown"]["dot_product_ms"],
        })

        print(f"[{i:02d}] {tc['label']:<14} | Mozc: {tc['mozc_top']} | Expected: {tc['expected']}")
        print(f"     Cross-Encoder: {c_winner} {'[OK]' if c_ok else '[NG]'} ({c_lat:5.1f} ms)")
        print(f"     Dual-Encoder : {d_winner} {'[OK]' if d_ok else '[NG]'} ({d_res['latency_ms']:5.1f} ms, dot-only: {d_res['breakdown']['dot_product_ms']:5.3f} ms)")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Mozc Baseline Accuracy        : 0.0%  (0/{len(BENCHMARK_CASES)}) - All homophone traps")
    print(f"Cross-Encoder (Current) Acc   : {cross_correct/len(BENCHMARK_CASES)*100:.1f}% ({cross_correct}/{len(BENCHMARK_CASES)})")
    print(f"Dual-Encoder (New) Acc        : {dual_correct/len(BENCHMARK_CASES)*100:.1f}% ({dual_correct}/{len(BENCHMARK_CASES)})")
    print("-" * 80)
    print(f"Cross-Encoder Mean Latency    : {statistics.mean(cross_latencies):.2f} ms (p50: {statistics.median(cross_latencies):.2f} ms)")
    print(f"Dual-Encoder Total Latency    : {statistics.mean(dual_latencies):.2f} ms (p50: {statistics.median(dual_latencies):.2f} ms) -> {statistics.mean(cross_latencies)/statistics.mean(dual_latencies):.1f}x faster!")
    print(f"Dual-Encoder Dot-Product Only : {statistics.mean(dual_dot_latencies):.4f} ms (p50: {statistics.median(dual_dot_latencies):.4f} ms) -> {statistics.mean(cross_latencies)/statistics.mean(dual_dot_latencies):.0f}x faster!")
    print("=" * 80)

    # Save comparison report
    report_path = ROOT / "build" / "dual_encoder_comparison_report.json"
    report_data = {
        "summary": {
            "cases": len(BENCHMARK_CASES),
            "cross_accuracy": cross_correct / len(BENCHMARK_CASES) * 100.0,
            "cross_mean_ms": round(statistics.mean(cross_latencies), 2),
            "dual_accuracy": dual_correct / len(BENCHMARK_CASES) * 100.0,
            "dual_mean_ms": round(statistics.mean(dual_latencies), 2),
            "dual_dot_only_ms": round(statistics.mean(dual_dot_latencies), 4),
            "speedup_ratio": round(statistics.mean(cross_latencies) / statistics.mean(dual_latencies), 1),
            "dot_speedup_ratio": round(statistics.mean(cross_latencies) / statistics.mean(dual_dot_latencies), 0),
        },
        "details": results,
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detailed comparison saved to {report_path}")


if __name__ == "__main__":
    run_benchmark()
