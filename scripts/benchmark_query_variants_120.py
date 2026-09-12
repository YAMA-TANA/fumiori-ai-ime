"""Compare the three ONNX query layouts on the fixed 120-question holdout."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker


def _holdout() -> List[Dict[str, Any]]:
    from scripts.create_and_evaluate_holdout_120 import HOLDOUT_TEST_SET

    if len(HOLDOUT_TEST_SET) != 120:
        raise AssertionError(f"expected 120 holdout questions, got {len(HOLDOUT_TEST_SET)}")
    return HOLDOUT_TEST_SET


def _request(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "request_id": f"query-variant-{index:03d}",
        "inference_trigger": "explicit",
        "segments": [{
            "id": "s0",
            "preceding_text": item["prefix"],
            "following_text": item["suffix"],
            "read": item["reading"],
            "candidates": [
                {"id": f"c{candidate_index}", "text": text,
                 "rank": candidate_index + 1}
                for candidate_index, text in enumerate(item["candidates"])
            ],
        }],
    }


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int((len(ordered) * percentile + 0.999999) - 1)))
    return ordered[index]


def evaluate(model_paths: List[Path], variant: str,
             execution_providers: List[str] | None = None) -> Dict[str, Any]:
    ranker = OnnxRuriReranker(
        settings={
            "compute_mode": "auto" if execution_providers else "cpu",
            "context_enabled": True,
            "context_chars": 128,
            "document_domain": "general",
            "custom_instruction": "",
            "lexical_grounding": True,
        },
        model_paths=model_paths,
        query_variant=variant,
        execution_providers=execution_providers,
    )
    latencies: List[float] = []
    correct = 0
    details = []
    for index, item in enumerate(_holdout(), start=1):
        request = _request(item, index)
        started = time.perf_counter()
        response = ranker.rank_batch(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        winner_id = response["segments"][0]["winner_id"]
        winner_index = int(winner_id[1:])
        winner = item["candidates"][winner_index]
        ok = winner == item["expected"]
        correct += int(ok)
        details.append({
            "id": item["id"],
            "winner": winner,
            "expected": item["expected"],
            "confidence": response["segments"][0]["confidence"],
            "latency_ms": round(latency_ms, 3),
            "correct": ok,
        })
    return {
        "variant": variant,
        "models": [str(path) for path in model_paths],
        "total": len(details),
        "correct": correct,
        "accuracy": round(correct / len(details) * 100.0, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=None,
        help="repeat for an ensemble; defaults to the single 70M INT8 model",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="force an ONNX Runtime provider, e.g. DmlExecutionProvider",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "query_variant_benchmark_120.json",
    )
    args = parser.parse_args()
    models = [
        path.resolve() for path in (args.model or [
            ROOT / "build" / "onnx-model-70m" / "ruri-ime-int8.onnx"
        ])
    ]
    missing = [str(path) for path in models if not path.is_file()]
    if missing:
        raise SystemExit(f"model not found: {', '.join(missing)}")
    providers = list(args.provider) if args.provider else None
    results = [
        evaluate(models, variant, providers)
        for variant in ("current", "short", "none")
    ]
    payload = {
        "questions": 120,
        "models": [str(path) for path in models],
        "execution_providers": providers,
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        [{
            "variant": result["variant"],
            "accuracy": f"{result['correct']}/{result['total']} ({result['accuracy']:.2f}%)",
            **result["latency_ms"],
        } for result in results],
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
