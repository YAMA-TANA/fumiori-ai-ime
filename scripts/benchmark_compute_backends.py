"""Compare ONNX Runtime CUDA and DirectML on the fixed 120-question holdout."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import onnxruntime as ort

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
        "request_id": f"compute-backend-{index:03d}",
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


def evaluate(model_paths: List[Path], provider: str) -> Dict[str, Any]:
    available = list(ort.get_available_providers())
    if provider not in available:
        return {
            "provider": provider,
            "status": "unavailable",
            "available_providers": available,
        }

    started = time.perf_counter()
    try:
        ranker = OnnxRuriReranker(
            settings={
                "compute_mode": "cpu",
                "context_enabled": True,
                "context_chars": 128,
                "document_domain": "general",
                "custom_instruction": "",
                "lexical_grounding": True,
            },
            model_paths=model_paths,
            query_variant="current",
            execution_providers=[provider, "CPUExecutionProvider"],
        )
    except Exception as exc:
        return {
            "provider": provider,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "available_providers": available,
        }

    latencies: List[float] = []
    correct = 0
    for index, item in enumerate(_holdout(), start=1):
        request = _request(item, index)
        request_started = time.perf_counter()
        response = ranker.rank_batch(request)
        latencies.append((time.perf_counter() - request_started) * 1000.0)
        winner_id = response["segments"][0]["winner_id"]
        winner = item["candidates"][int(winner_id[1:])]
        correct += int(winner == item["expected"])

    return {
        "provider": provider,
        "status": "ok",
        "device": ranker.device,
        "active_providers": ranker.sessions[0].get_providers(),
        "models": [str(path) for path in model_paths],
        "load_and_warmup_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "total": len(latencies),
        "correct": correct,
        "accuracy": round(correct / len(latencies) * 100.0, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=None,
        help="repeat for an ensemble; defaults to the single 70M FP16 model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "compute_backend_benchmark_120.json",
    )
    args = parser.parse_args()
    models = [
        path.resolve() for path in (args.model or [
            ROOT / "build" / "onnx-model-70m" / "ruri-ime-fp16.onnx"
        ])
    ]
    missing = [str(path) for path in models if not path.is_file()]
    if missing:
        raise SystemExit(f"model not found: {', '.join(missing)}")

    results = [
        evaluate(models, provider)
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider")
    ]
    payload = {
        "questions": 120,
        "models": [str(path) for path in models],
        "available_providers": list(ort.get_available_providers()),
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
