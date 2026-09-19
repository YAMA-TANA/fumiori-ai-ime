"""Evaluate two independently merged LoRA 70M ONNX models as a score ensemble.

Each model sees the same request, and candidate evidence scores are averaged
before selecting the top candidate.  This keeps the comparison deterministic
and makes the complementary residual LoRA rounds reproducible without
modifying the production single-model path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from create_and_evaluate_holdout_120 import HOLDOUT_TEST_SET, verify_strictly_unseen
from ranker.onnx_ranker import OnnxRuriReranker

ROOT = Path(__file__).resolve().parents[1]


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


def evaluate(model_a: Path, model_b: Path, output: Path) -> Dict[str, Any]:
    verify_strictly_unseen(HOLDOUT_TEST_SET)
    rankers = [_ranker(model_a), _ranker(model_b)]
    details = []
    correct = 0
    for q in HOLDOUT_TEST_SET:
        req = {
            "request_id": f"ensemble-{q['id']}",
            "preceding_text": q["prefix"],
            "following_text": q["suffix"],
            "read": q["reading"],
            "candidates": [
                {"id": f"c{i}", "text": text, "rank": i}
                for i, text in enumerate(q["candidates"])
            ],
        }
        explanations = []
        for ranker in rankers:
            ranker.rank(req)
            explanations.append({item["id"]: item for item in ranker.last_explanation["candidates"]})
        candidate_ids = explanations[0].keys()
        ensemble_weights = (0.50, 0.50) if len(q["reading"]) <= 3 else (0.25, 0.75)
        averaged = {
            cid: sum(
                weight * explanation[cid]["evidence_score"]
                for weight, explanation in zip(ensemble_weights, explanations)
            )
            for cid in candidate_ids
        }
        selected_id = max(averaged, key=averaged.get)
        selected = next(item["text"] for item in req["candidates"] if item["id"] == selected_id)
        ok = selected == q["expected"]
        correct += int(ok)
        details.append(
            {
                "id": q["id"],
                "expected": q["expected"],
                "selected": selected,
                "ok": ok,
                "average_evidence_scores": averaged,
            }
        )
    result = {
        "model_a": str(model_a),
        "model_b": str(model_b),
        "total": len(HOLDOUT_TEST_SET),
        "correct": correct,
        "accuracy": round(correct / len(HOLDOUT_TEST_SET) * 100.0, 2),
        "ensemble_weights": "short readings 50/50; other readings 25/75",
        "details": details,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model_a", "model_b", "total", "correct", "accuracy")}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "holdout_120_lora_ensemble.json")
    args = parser.parse_args()
    evaluate(args.model_a.resolve(), args.model_b.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
