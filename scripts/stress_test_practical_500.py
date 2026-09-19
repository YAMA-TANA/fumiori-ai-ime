"""Execute 500-question practical stress test for Dual-Encoder IME Reranker."""

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


def run_500_stress_test() -> dict:
    print("=" * 80)
    print("PRACTICAL 500-QUESTION FULL STRESS TEST (DUAL-ENCODER 70M)")
    print("=" * 80)

    dataset_path = ROOT / "build" / "practical_500_testset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Testset not found at {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} test cases from {dataset_path.name}.")

    model_dir = ROOT / "build" / "dual-encoder-70m"
    ranker = DualEncoderIMEReranker(model_dir)

    # 1. Preload candidate cache
    all_words = set()
    for item in cases:
        for c in item.get("candidates", []):
            all_words.add(c["text"])
    ranker.preload_candidates(list(all_words))
    print(f"Preloaded {len(all_words)} unique candidate vectors into memory cache.")

    # 2. Warmup
    warmup_req = {
        "request_id": "warmup",
        "preceding_text": "テスト文脈",
        "following_text": "",
        "read": "てすと",
        "candidates": [{"id": "c1", "text": "テスト", "rank": 1}],
    }
    ranker.rank(warmup_req)

    # 3. Run evaluation
    results = []
    latencies_full = []
    latencies_dot = []

    group_stats = defaultdict(lambda: {"total": 0, "correct": 0, "mozc_correct": 0, "recoveries": 0, "regressions": 0})
    subgroup_stats = defaultdict(lambda: {"total": 0, "correct": 0, "mozc_correct": 0, "recoveries": 0, "regressions": 0})
    forbidden_rare_violations = []

    for idx, item in enumerate(cases, start=1):
        req = {
            "request_id": item["id"],
            "preceding_text": item.get("prefix", ""),
            "following_text": item.get("suffix", ""),
            "read": item.get("reading", ""),
            "candidates": item["candidates"],
        }

        res = ranker.rank(req)
        winner_text = res["winner_text"]
        acceptable = set(item.get("acceptable", [item.get("expected", "")]))
        expected = item.get("expected", "")

        is_correct = (winner_text in acceptable)
        
        # Mozc raw rank 1 candidate
        mozc_raw_1 = item["candidates"][0]["text"] if item["candidates"] else ""
        mozc_is_correct = (mozc_raw_1 in acceptable)

        is_recovery = (not mozc_is_correct and is_correct)
        is_regression = (mozc_is_correct and not is_correct)

        # Check rare word violation (e.g. 江青 in vague/zero context)
        forbidden_rare = item.get("forbidden_rare")
        if forbidden_rare and winner_text == forbidden_rare:
            forbidden_rare_violations.append({
                "id": item["id"],
                "prefix": item.get("prefix", ""),
                "reading": item.get("reading", ""),
                "winner": winner_text,
                "forbidden": forbidden_rare,
            })

        lat_full = res["latency_ms"]
        lat_dot = res["breakdown"]["dot_product_ms"]
        latencies_full.append(lat_full)
        latencies_dot.append(lat_dot)

        # Update stats
        grp = item.get("group", "general")
        subgrp = item.get("subgroup", grp)

        group_stats[grp]["total"] += 1
        group_stats[grp]["correct"] += int(is_correct)
        group_stats[grp]["mozc_correct"] += int(mozc_is_correct)
        group_stats[grp]["recoveries"] += int(is_recovery)
        group_stats[grp]["regressions"] += int(is_regression)

        subgroup_stats[subgrp]["total"] += 1
        subgroup_stats[subgrp]["correct"] += int(is_correct)
        subgroup_stats[subgrp]["mozc_correct"] += int(mozc_is_correct)
        subgroup_stats[subgrp]["recoveries"] += int(is_recovery)
        subgroup_stats[subgrp]["regressions"] += int(is_regression)

        results.append({
            "id": item["id"],
            "group": grp,
            "subgroup": subgrp,
            "reading": item.get("reading", ""),
            "prefix": item.get("prefix", ""),
            "expected": expected,
            "mozc_top": mozc_raw_1,
            "winner": winner_text,
            "is_correct": is_correct,
            "mozc_correct": mozc_is_correct,
            "is_recovery": is_recovery,
            "is_regression": is_regression,
            "latency_ms": lat_full,
            "dot_ms": lat_dot,
        })

    # Summary calculations
    total_cases = len(cases)
    total_correct = sum(1 for r in results if r["is_correct"])
    total_mozc_correct = sum(1 for r in results if r["mozc_correct"])
    total_recoveries = sum(1 for r in results if r["is_recovery"])
    total_regressions = sum(1 for r in results if r["is_regression"])

    overall_acc = (total_correct / total_cases) * 100.0
    mozc_acc = (total_mozc_correct / total_cases) * 100.0

    print("\n" + "=" * 80)
    print("500-QUESTION STRESS TEST OVERALL RESULTS")
    print("=" * 80)
    print(f"Total Test Cases            : {total_cases}")
    print(f"Mozc Default Accuracy       : {mozc_acc:.2f}% ({total_mozc_correct}/{total_cases})")
    print(f"Dual-Encoder 70M Accuracy   : {overall_acc:.2f}% ({total_correct}/{total_cases})")
    print(f"Net Accuracy Gain           : +{overall_acc - mozc_acc:.2f}%")
    print(f"Mozc Errors Recovered       : {total_recoveries} cases")
    print(f"Regressions (Mozc OK -> NG) : {total_regressions} cases")
    print(f"Forbidden Rare Word Leaks   : {len(forbidden_rare_violations)} cases (e.g. '江青' violations)")

    print("\n" + "-" * 80)
    print("PERFORMANCE & LATENCY PROFILE")
    print("-" * 80)
    print(f"Full Latency (Mean)         : {statistics.mean(latencies_full):.2f} ms")
    print(f"Full Latency (Median / p50) : {statistics.median(latencies_full):.2f} ms")
    print(f"Full Latency (p95)          : {np.percentile(latencies_full, 95):.2f} ms")
    print(f"Full Latency (p99)          : {np.percentile(latencies_full, 99):.2f} ms")
    print(f"Full Latency (Max)          : {max(latencies_full):.2f} ms")
    print(f"Vector Dot-Product (Mean)   : {statistics.mean(latencies_dot):.4f} ms ({statistics.mean(latencies_dot)*1000:.1f} microseconds)")
    print(f"Vector Dot-Product (p50)    : {statistics.median(latencies_dot):.4f} ms")

    print("\n" + "-" * 80)
    print("BREAKDOWN BY CATEGORY & REAL-WORLD SCENARIO")
    print("-" * 80)
    for grp, stats in group_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100.0
        m_acc = (stats["mozc_correct"] / stats["total"]) * 100.0
        print(f"[{grp}] {stats['correct']}/{stats['total']} ({acc:.1f}%) | Mozc: {m_acc:.1f}% | Rec: {stats['recoveries']}, Reg: {stats['regressions']}")

    print("\n" + "-" * 80)
    print("SUBGROUP DETAILS")
    print("-" * 80)
    for subgrp, stats in subgroup_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100.0
        m_acc = (stats["mozc_correct"] / stats["total"]) * 100.0
        print(f"  - {subgrp:35s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%) [Mozc: {m_acc:5.1f}% | Rec: {stats['recoveries']:2d}, Reg: {stats['regressions']:2d}]")

    report = {
        "total_cases": total_cases,
        "dual_encoder_accuracy": round(overall_acc, 2),
        "mozc_accuracy": round(mozc_acc, 2),
        "accuracy_gain": round(overall_acc - mozc_acc, 2),
        "total_recoveries": total_recoveries,
        "total_regressions": total_regressions,
        "rare_word_violations_count": len(forbidden_rare_violations),
        "rare_word_violations": forbidden_rare_violations,
        "latency_stats": {
            "mean_full_ms": round(statistics.mean(latencies_full), 2),
            "p50_full_ms": round(statistics.median(latencies_full), 2),
            "p95_full_ms": round(float(np.percentile(latencies_full, 95)), 2),
            "p99_full_ms": round(float(np.percentile(latencies_full, 99)), 2),
            "max_full_ms": round(max(latencies_full), 2),
            "mean_dot_ms": round(statistics.mean(latencies_dot), 4),
            "p50_dot_ms": round(statistics.median(latencies_dot), 4),
        },
        "group_stats": dict(group_stats),
        "subgroup_stats": dict(subgroup_stats),
        "rows": results,
    }

    out_json = ROOT / "build" / "practical_500_stress_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[SUCCESS] Full 500-question stress report saved to {out_json}")
    return report


if __name__ == "__main__":
    run_500_stress_test()
