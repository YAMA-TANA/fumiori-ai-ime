"""Compare calibrated LoRA3/LoRA6 ensemble selection policies.

This is an offline study tool.  It never changes the candidate set: each
policy only selects one of the surfaces supplied by the benchmark request.
The useful distinction here is whether the two models' evidence should be
averaged in raw-score space or normalized per request before voting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker


def _ranker(path: Path) -> OnnxRuriReranker:
    return OnnxRuriReranker(
        settings={
            "compute_mode": "cpu",
            "context_enabled": True,
            "context_chars": 128,
            "document_domain": "general",
            "custom_instruction": "",
            "lexical_grounding": True,
        },
        model_path=path,
    )


def _request(item: Mapping[str, Any], request_id: str) -> Dict[str, Any]:
    prefix = str(item.get("prefix", item.get("context", "")))
    suffix = str(item.get("suffix", ""))
    reading = str(item.get("reading", item.get("read", "")))
    candidates = list(item["candidates"])
    return {
        "request_id": request_id,
        "preceding_text": prefix,
        "following_text": suffix,
        "read": reading,
        "candidates": [
            {"id": f"c{i}", "text": text, "rank": i}
            for i, text in enumerate(candidates)
        ],
    }


def _load_cases() -> Dict[str, List[Dict[str, Any]]]:
    from create_and_evaluate_holdout_120 import HOLDOUT_TEST_SET
    from evaluate_new_holdout import NEW_HOLDOUT
    from evaluate_preceding_only_range_holdout import CASES

    strict = []
    for item in HOLDOUT_TEST_SET:
        strict.append({**item, "prefix": item["prefix"], "suffix": item["suffix"]})
    fresh = []
    for item in NEW_HOLDOUT:
        fresh.append({**item, "prefix": item["prefix"], "suffix": item.get("suffix", "")})
    preceding = []
    for item in CASES:
        preceding.append({**item, "prefix": item["prefix"], "suffix": ""})
    return {"strict120": strict, "fresh100": fresh, "preceding152": preceding}


def _evidence(ranker: OnnxRuriReranker, request: Dict[str, Any]) -> Dict[str, float]:
    ranker.rank(request)
    return {
        str(item["id"]): float(item["evidence_score"])
        for item in ranker.last_explanation["candidates"]
    }


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    scale = max(float(temperature), 1e-6)
    top = max(values)
    exps = [math.exp((value - top) / scale) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _zscore(values: Sequence[float]) -> List[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation < 1e-6:
        return [0.0] * len(values)
    return [(value - mean) / deviation for value in values]


def _policies() -> Dict[str, Callable[[List[Dict[str, float]]], str]]:
    policies: Dict[str, Callable[[List[Dict[str, float]]], str]] = {}

    def raw_mean(scores: List[Dict[str, float]]) -> str:
        return max(scores[0], key=lambda cid: sum(score[cid] for score in scores) / len(scores))

    policies["raw_mean"] = raw_mean
    for weight in (0.25, 0.40, 0.50, 0.60, 0.75):
        def weighted(scores: List[Dict[str, float]], weight: float = weight) -> str:
            return max(scores[0], key=lambda cid: weight * scores[0][cid] + (1.0 - weight) * scores[1][cid])
        policies[f"raw_a{weight:.2f}"] = weighted

    for temperature in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00):
        def probability(scores: List[Dict[str, float]], temperature: float = temperature) -> str:
            ids = list(scores[0])
            pa = _softmax([scores[0][cid] for cid in ids], temperature)
            pb = _softmax([scores[1][cid] for cid in ids], temperature)
            return ids[max(range(len(ids)), key=lambda index: (pa[index] + pb[index]) / 2.0)]
        policies[f"softmax_t{temperature:.2f}"] = probability

    def normalized(scores: List[Dict[str, float]]) -> str:
        ids = list(scores[0])
        za = _zscore([scores[0][cid] for cid in ids])
        zb = _zscore([scores[1][cid] for cid in ids])
        return ids[max(range(len(ids)), key=lambda index: za[index] + zb[index])]

    policies["zscore_mean"] = normalized

    def rank_vote(scores: List[Dict[str, float]]) -> str:
        ids = list(scores[0])
        ranks = []
        for score in scores:
            ordered = sorted(ids, key=lambda cid: score[cid], reverse=True)
            ranks.append({cid: rank for rank, cid in enumerate(ordered)})
        return min(ids, key=lambda cid: ranks[0][cid] + ranks[1][cid])

    policies["rank_vote"] = rank_vote
    return policies


def evaluate(model_a: Path, model_b: Path, output: Path) -> Dict[str, Any]:
    rankers = [_ranker(model_a), _ranker(model_b)]
    runtime_ensemble = OnnxRuriReranker(
        settings={
            "compute_mode": "cpu",
            "context_enabled": True,
            "context_chars": 128,
            "document_domain": "general",
            "custom_instruction": "",
            "lexical_grounding": True,
        },
        model_paths=[model_a, model_b],
        prior_w=0.0,
        safety_gate=False,
    )
    policies = _policies()
    result: Dict[str, Any] = {name: {} for name in _load_cases()}
    for corpus, cases in _load_cases().items():
        counts = {name: 0 for name in policies}
        dynamic_short_counts = 0
        runtime_correct = 0
        group_counts: Dict[str, Dict[str, int]] = {}
        policy_differences: List[Dict[str, Any]] = []
        for index, item in enumerate(cases, start=1):
            request = _request(item, f"policy-{corpus}-{index:03d}")
            scores = [_evidence(ranker, request) for ranker in rankers]
            expected = str(item["expected"])
            expected_id = next(
                (str(candidate["id"]) for candidate in request["candidates"]
                 if candidate["text"] == expected),
                None,
            )
            if expected_id is None:
                continue
            runtime_response = runtime_ensemble.rank(request)
            runtime_correct += int(runtime_response["candidates"][0]["id"] == expected_id)
            for name, policy in policies.items():
                counts[name] += int(policy(scores) == expected_id)
            dynamic_weights = (0.50, 0.50) if len(request["read"]) <= 3 else (0.25, 0.75)
            dynamic_scores = {
                cid: sum(
                    weight * score[cid]
                    for weight, score in zip(dynamic_weights, scores)
                )
                for cid in scores[0]
            }
            dynamic_short_counts += int(
                max(dynamic_scores, key=dynamic_scores.get) == expected_id
            )
            group = str(item.get("group", item.get("category", "all")))
            group_counts.setdefault(group, {name: 0 for name in policies})
            for name, policy in policies.items():
                group_counts[group][name] += int(policy(scores) == expected_id)
            raw_id = policies["raw_mean"](scores)
            weighted_id = policies["raw_a0.25"](scores)
            if raw_id != weighted_id:
                policy_differences.append({
                    "index": index,
                    "expected": expected,
                    "raw_mean": next(c["text"] for c in request["candidates"] if c["id"] == raw_id),
                    "raw_a0.25": next(c["text"] for c in request["candidates"] if c["id"] == weighted_id),
                    "group": group,
                })
        result[corpus] = {
            "total": len(cases),
            "scores": {
                name: {"correct": correct, "accuracy": round(correct / len(cases) * 100.0, 2)}
                for name, correct in counts.items()
            },
            "dynamic_short_equal": {
                "correct": dynamic_short_counts,
                "accuracy": round(dynamic_short_counts / len(cases) * 100.0, 2),
            },
            "groups": group_counts,
            "raw_mean_vs_raw_a0.25": policy_differences,
            "runtime_weighted_gate": {
                "correct": runtime_correct,
                "accuracy": round(runtime_correct / len(cases) * 100.0, 2),
            },
        }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "ensemble_policy_evaluation.json")
    args = parser.parse_args()
    evaluate(args.model_a.resolve(), args.model_b.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
