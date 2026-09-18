"""Comprehensive Stress Test Suite for 70M Dual-Encoder IME Reranker."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List

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

import numpy as np
import torch
from ranker.dual_encoder_ranker import DualEncoderIMEReranker


def load_holdout_120() -> list[dict]:
    path = ROOT / "scripts" / "create_and_evaluate_holdout_120.py"
    spec = importlib.util.spec_from_file_location("holdout_120", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.HOLDOUT_TEST_SET)


def stress_test_1_holdout_120(ranker: DualEncoderIMEReranker) -> dict:
    """Stress Test 1: Full 120-Question Strict Holdout Benchmark."""
    print("\n" + "=" * 80)
    print("[STRESS TEST 1] 120-QUESTION STRICT HOLDOUT FULL EVALUATION")
    print("=" * 80)
    
    holdout = load_holdout_120()
    print(f"Loaded {len(holdout)} strict holdout questions.")
    
    # Pre-warm candidates
    all_cands = set()
    for item in holdout:
        all_cands.update(item["candidates"])
    ranker.preload_candidates(list(all_cands))
    print(f"Preloaded {len(all_cands)} candidate vectors.")

    correct = 0
    latencies = []
    dot_latencies = []
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    failures = []

    for item in holdout:
        req = {
            "request_id": item["id"],
            "preceding_text": item["prefix"],
            "following_text": item["suffix"],
            "read": item["reading"],
            "candidates": [
                {"id": f"c{idx}", "text": cand, "rank": idx + 1}
                for idx, cand in enumerate(item["candidates"])
            ]
        }
        res = ranker.rank(req)
        winner = res["winner_text"]
        ok = (winner == item["expected"])
        correct += int(ok)
        latencies.append(res["latency_ms"])
        dot_latencies.append(res["breakdown"]["dot_product_ms"])

        cat = item.get("category", "その他")
        category_stats[cat]["total"] += 1
        category_stats[cat]["correct"] += int(ok)

        if not ok:
            failures.append({
                "id": item["id"],
                "category": cat,
                "reading": item["reading"],
                "prefix": item["prefix"],
                "suffix": item["suffix"],
                "expected": item["expected"],
                "predicted": winner,
                "candidates": item["candidates"],
                "mozc_default": item["candidates"][0],
            })

    acc = correct / len(holdout) * 100.0
    print(f"Holdout 120 Accuracy       : {acc:.2f}% ({correct}/{len(holdout)})")
    print(f"Mean Latency (Full)        : {statistics.mean(latencies):.2f} ms (p50: {statistics.median(latencies):.2f} ms, max: {max(latencies):.2f} ms)")
    print(f"Mean Latency (Dot-Product) : {statistics.mean(dot_latencies):.4f} ms (p50: {statistics.median(dot_latencies):.4f} ms)")
    
    print("\n[Category Breakdown]")
    for cat, s in sorted(category_stats.items()):
        c_acc = s["correct"] / s["total"] * 100.0
        print(f"  - {cat:<12}: {s['correct']}/{s['total']} ({c_acc:.1f}%)")

    return {
        "total": len(holdout),
        "correct": correct,
        "accuracy": acc,
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "dot_latency_mean_ms": round(statistics.mean(dot_latencies), 4),
        "category_breakdown": {k: {"total": v["total"], "correct": v["correct"], "accuracy": round(v["correct"]/v["total"]*100, 1)} for k, v in category_stats.items()},
        "failures_count": len(failures),
        "failures_sample": failures[:10],
    }


def stress_test_2_full_database_candidates(ranker: DualEncoderIMEReranker) -> dict:
    """Stress Test 2: Large Candidate Lists (15-60 candidates per request)."""
    print("\n" + "=" * 80)
    print("[STRESS TEST 2] FULL DATABASE CANDIDATE STRESS TEST (Huge Candidate Lists)")
    print("=" * 80)

    db_path = ROOT / "data" / "massive_homophone_database.json"
    if not db_path.exists():
        print(f"Database {db_path} not found, skipping.")
        return {}

    database = json.loads(db_path.read_text(encoding="utf-8"))
    holdout = load_holdout_120()

    stress_cases = []
    for item in holdout:
        entry = database.get(item["reading"])
        if entry and item["expected"] in entry["candidates"]:
            cand_list = list(entry["candidates"])
            if len(cand_list) >= 10:  # Only test heavy candidate sets (10 to 60+ candidates)
                stress_cases.append((item, cand_list))

    print(f"Found {len(stress_cases)} heavy candidate cases with 10+ candidates each.")
    
    # Pre-cache
    all_words = set()
    for _, cands in stress_cases:
        all_words.update(cands)
    ranker.preload_candidates(list(all_words))
    print(f"Preloaded {len(all_words)} candidates into memory cache.")

    correct = 0
    latencies = []
    dot_latencies = []
    cand_counts = []

    for item, cands in stress_cases:
        cand_counts.append(len(cands))
        req = {
            "request_id": f"stress-db-{item['id']}",
            "preceding_text": item["prefix"],
            "read": item["reading"],
            "candidates": [{"id": f"c{i}", "text": c, "rank": i+1} for i, c in enumerate(cands)]
        }
        res = ranker.rank(req)
        ok = (res["winner_text"] == item["expected"])
        correct += int(ok)
        latencies.append(res["latency_ms"])
        dot_latencies.append(res["breakdown"]["dot_product_ms"])

    acc = correct / max(1, len(stress_cases)) * 100.0
    print(f"Heavy Candidate Cases Evaluated: {len(stress_cases)}")
    print(f"Average Candidate Count        : {statistics.mean(cand_counts):.1f} candidates (max: {max(cand_counts)})")
    print(f"Accuracy with Full Candidate DB: {acc:.2f}% ({correct}/{len(stress_cases)})")
    print(f"Mean Full Latency              : {statistics.mean(latencies):.2f} ms")
    print(f"Mean Dot-Product Latency       : {statistics.mean(dot_latencies):.4f} ms (p50: {statistics.median(dot_latencies):.4f} ms)")

    return {
        "cases": len(stress_cases),
        "mean_candidates": round(statistics.mean(cand_counts), 1),
        "max_candidates": max(cand_counts),
        "accuracy": acc,
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "dot_latency_mean_ms": round(statistics.mean(dot_latencies), 4),
    }


def stress_test_3_context_lengths(ranker: DualEncoderIMEReranker) -> dict:
    """Stress Test 3: Context Length Sensitivity (0, 8, 32, 64, 128, 256 chars)."""
    print("\n" + "=" * 80)
    print("[STRESS TEST 3] CONTEXT LENGTH SENSITIVITY & ROBUSTNESS")
    print("=" * 80)

    base_prefix = "当社では昨年度より全社的な業務効率化プロジェクトを推進しており、先週開催された経営会議において役員が稟議書を"
    reading = "けっさい"
    candidates = ["決済", "決裁", "けっさい", "決裁者", "血清"]
    expected = "決裁"

    lengths = [0, 4, 8, 16, 32, 64, 100]
    results = {}

    for l in lengths:
        sub_prefix = base_prefix[-l:] if l > 0 else ""
        req = {
            "request_id": f"ctx-len-{l}",
            "preceding_text": sub_prefix,
            "read": reading,
            "candidates": [{"id": f"c{i}", "text": c, "rank": i+1} for i, c in enumerate(candidates)]
        }
        res = ranker.rank(req)
        winner = res["winner_text"]
        scores = {c.get("text", c.get("word", "")): round(c.get("dot_score", 0.0), 4) for c in res["candidates"]}
        print(f"Context Length: {l:3d} chars | Prefix: '...{sub_prefix[-20:]}'")
        print(f"  -> Winner: '{winner}' (Expected: '{expected}') | Dot-Time: {res['breakdown']['dot_product_ms']:.4f} ms")
        print(f"  -> Scores: {scores}")
        results[f"len_{l}"] = {
            "length": l,
            "winner": winner,
            "correct": winner == expected,
            "scores": scores,
            "latency_ms": res["latency_ms"],
            "dot_ms": res["breakdown"]["dot_product_ms"],
        }

    return results


def stress_test_4_continuous_throughput(ranker: DualEncoderIMEReranker, iterations: int = 500) -> dict:
    """Stress Test 4: High-Throughput Burst & Memory Stability (500 continuous conversions)."""
    print("\n" + "=" * 80)
    print(f"[STRESS TEST 4] CONTINUOUS BURST & THROUGHPUT STRESS ({iterations} CONVERSIONS)")
    print("=" * 80)

    req = {
        "request_id": "burst-test",
        "preceding_text": "患者の容態を安定させるため適切な薬で病気を",
        "read": "なおす",
        "candidates": [
            {"id": "c1", "text": "直す", "rank": 1},
            {"id": "c2", "text": "治す", "rank": 2},
            {"id": "c3", "text": "猶す", "rank": 3},
            {"id": "c4", "text": "なおす", "rank": 4},
        ]
    }

    # Warmup
    for _ in range(10):
        ranker.rank(req)

    latencies = []
    dot_latencies = []
    t_start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        res = ranker.rank(req)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        dot_latencies.append(res["breakdown"]["dot_product_ms"])

    total_time = time.perf_counter() - t_start
    qps = iterations / total_time

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile

    dot_p50 = statistics.median(dot_latencies)
    dot_p99 = statistics.quantiles(dot_latencies, n=100)[98]

    print(f"Completed {iterations} conversions in {total_time:.2f} s")
    print(f"Throughput (QPS)          : {qps:.1f} requests/sec")
    print(f"Full Latency Distribution : Mean: {statistics.mean(latencies):.2f} ms | p50: {p50:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms | Max: {max(latencies):.2f} ms")
    print(f"Dot-Product Distribution  : Mean: {statistics.mean(dot_latencies):.4f} ms | p50: {dot_p50:.4f} ms | p99: {dot_p99:.4f} ms")

    # Check GPU memory if available
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"PyTorch CUDA VRAM Allocated: {vram_mb:.2f} MB (Extremely compact!)")

    return {
        "iterations": iterations,
        "total_time_s": round(total_time, 2),
        "qps": round(qps, 1),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max(latencies), 2),
        "dot_mean_ms": round(statistics.mean(dot_latencies), 4),
        "dot_p50_ms": round(dot_p50, 4),
        "dot_p99_ms": round(dot_p99, 4),
        "vram_allocated_mb": round(vram_mb, 2),
    }


def main():
    model_dir = ROOT / "build" / "dual-encoder-70m"
    print(f"Initializing DualEncoderIMEReranker from {model_dir}...")
    ranker = DualEncoderIMEReranker(model_dir)

    test1_res = stress_test_1_holdout_120(ranker)
    test2_res = stress_test_2_full_database_candidates(ranker)
    test3_res = stress_test_3_context_lengths(ranker)
    test4_res = stress_test_4_continuous_throughput(ranker, iterations=500)

    # Save full stress report
    out_path = ROOT / "build" / "dual_encoder_full_stress_report.json"
    report = {
        "test1_holdout_120": test1_res,
        "test2_heavy_candidates": test2_res,
        "test3_context_sensitivity": test3_res,
        "test4_throughput_burst": test4_res,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Full stress test report saved to {out_path}")


if __name__ == "__main__":
    main()
