"""Search conservative ensemble safety-gate profiles on existing holdouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker
from ranker.scoring import preserve_mozc_top_if_uncertain, rank_prior_penalty
from scripts.evaluate_ensemble_policies import _load_cases, _ranker, _request


PROFILES: Dict[str, Mapping[str, float] | None] = {
    "current": None,
    "ensemble_moderate": {
        "minimum_switch_delta": 0.25,
        "minimum_switch_margin": 0.10,
        "contextual_neural_confidence": 0.25,
        "contextual_lead_over_mozc": 1.20,
    },
    "ensemble_balanced": {
        "minimum_switch_delta": 0.35,
        "minimum_switch_margin": 0.15,
        "contextual_neural_confidence": 0.25,
        "contextual_lead_over_mozc": 1.40,
    },
    "ensemble_strong": {
        "minimum_switch_delta": 0.25,
        "minimum_switch_margin": 0.00,
        "contextual_neural_confidence": 0.20,
        "contextual_lead_over_mozc": 0.80,
    },
    "no_gate": {},
}


def evaluate(model_a: Path, model_b: Path, output: Path, prior_w: float) -> dict:
    rankers = [_ranker(model_a), _ranker(model_b)]
    results: Dict[str, Any] = {name: {} for name in PROFILES}
    for corpus, cases in _load_cases().items():
        counters = {
            name: {"correct": 0, "total": len(cases), "stress": 0, "control": 0}
            for name in PROFILES
        }
        for index, item in enumerate(cases, start=1):
            request = _request(item, f"gate-{corpus}-{index:03d}")
            explanations = []
            for ranker in rankers:
                ranker.rank(request)
                explanations.append({x["id"]: x for x in ranker.last_explanation["candidates"]})
            ids = [str(candidate["id"]) for candidate in request["candidates"]]
            evidence = {
                cid: 0.25 * explanations[0][cid]["evidence_score"]
                + 0.75 * explanations[1][cid]["evidence_score"]
                for cid in ids
            }
            text_by_id = {str(candidate["id"]): str(candidate["text"]) for candidate in request["candidates"]}
            base = sorted(
                [
                    (evidence[cid] - rank_prior_penalty(index, prior_w), index, cid)
                    for index, cid in enumerate(ids)
                ],
                reverse=True,
            )
            expected_ids = {
                str(candidate["id"])
                for candidate in request["candidates"]
                if candidate["text"] == item["expected"]
            }
            group = str(item.get("group", "all"))
            for name, thresholds in PROFILES.items():
                if name == "no_gate":
                    selected = base
                else:
                    selected = preserve_mozc_top_if_uncertain(
                        base,
                        request["preceding_text"],
                        request["following_text"],
                        evidence,
                        text_by_id,
                        request["read"],
                        thresholds,
                    )
                ok = selected[0][2] in expected_ids
                counters[name]["correct"] += int(ok)
                counters[name][group] = counters[name].get(group, 0) + int(ok)
        for name, counter in counters.items():
            result = {
                "correct": counter["correct"],
                "total": counter["total"],
                "accuracy": round(counter["correct"] / counter["total"] * 100.0, 2),
            }
            if "stress" in counter:
                result["stress_correct"] = counter["stress"]
                result["control_correct"] = counter["control"]
            results[name][corpus] = result
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--prior-w", type=float, default=0.22)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "ensemble_gate_search.json")
    args = parser.parse_args()
    evaluate(args.model_a.resolve(), args.model_b.resolve(), args.output.resolve(), args.prior_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
