"""Evaluate the production-shaped multi-ONNX ranker path on local holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker
from scripts.evaluate_ensemble_policies import _load_cases, _request


def evaluate(
    model_a: Path,
    model_b: Path,
    output: Path,
    *,
    safety_gate: bool | None,
    prior_w: float | None,
) -> dict:
    ranker = OnnxRuriReranker(
        settings={
            "compute_mode": "cpu",
            "context_enabled": True,
            "context_chars": 128,
            "document_domain": "general",
            "custom_instruction": "",
            "lexical_grounding": True,
        },
        model_paths=[model_a, model_b],
        safety_gate=safety_gate,
        prior_w=prior_w,
    )
    result = {}
    for corpus, cases in _load_cases().items():
        correct = 0
        for index, item in enumerate(cases, start=1):
            request = _request(item, f"runtime-{corpus}-{index:03d}")
            response = ranker.rank(request)
            selected_id = response["candidates"][0]["id"]
            expected_ids = [
                candidate["id"]
                for candidate in request["candidates"]
                if candidate["text"] == item["expected"]
            ]
            correct += int(selected_id in expected_ids)
        result[corpus] = {
            "total": len(cases),
            "correct": correct,
            "accuracy": round(correct / len(cases) * 100.0, 2),
        }
    result["settings"] = {
        "safety_gate": safety_gate,
        "prior_w": prior_w,
        "weights": [0.25, 0.75],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument(
        "--safety-gate", action="store_true",
        help="force the conservative single-model gate for comparison",
    )
    parser.add_argument("--prior-w", type=float, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "runtime_ensemble_evaluation.json")
    args = parser.parse_args()
    evaluate(
        args.model_a.resolve(), args.model_b.resolve(), args.output.resolve(),
        safety_gate=True if args.safety_gate else None, prior_w=args.prior_w,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
