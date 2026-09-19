"""Evaluate IME conversion with only text to the left of the cursor.

The real conversion request has no reliable right context.  This benchmark
therefore always sends ``following_text == ""`` and varies only the conversion
range: lexical word, word plus particle, and an intentionally long single
segment containing a phrase such as ``製品の仕様を``.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker
from scripts.evaluate_practical_particle_holdout import BASE, CONTROL_BASE


def wide(reading: str, prefix: str, expected: str, candidates: List[str], category: str, control: bool = False) -> Dict[str, Any]:
    assert expected in candidates
    if control:
        assert candidates[0] == expected
    else:
        assert candidates[0] != expected
    return {
        "reading": reading,
        "prefix": prefix,
        "expected": expected,
        "candidates": candidates,
        "category": category,
        "control": control,
    }


WIDE_STRESS = [
    wide("せいひんのしようを", "顧客が", "製品の仕様を", ["製品の使用を", "製品の仕様を", "製品の試用を"], "長い一セグメント"),
    wide("こうつうひのせいさんを", "経理は", "交通費の精算を", ["交通費の生産を", "交通費の精算を", "交通費の清算を"], "長い一セグメント"),
    wide("けいやくしょのきていを", "担当者は", "契約書の規定を", ["契約書の規程を", "契約書の規定を", "契約書の既定を"], "長い一セグメント"),
    wide("けんきゅうろんぶんのかいていを", "編集部は", "研究論文の改訂を", ["研究論文の改定を", "研究論文の改訂を", "研究論文の開廷を"], "長い一セグメント"),
    wide("げんこうのこうせいを", "印刷所へ", "原稿の校正を", ["原稿の構成を", "原稿の校正を", "原稿の更正を"], "長い一セグメント"),
    wide("しすてむのしょうがいが", "運用中に", "システムの障害が", ["システムの傷害が", "システムの障害が", "システムの渉外が"], "長い一セグメント"),
    wide("けいやくのきかんを", "法務は", "契約の期間を", ["契約の機関を", "契約の期間を", "契約の器官を"], "長い一セグメント"),
    wide("ほしょうきんのへんかんを", "解約後に", "保証金の返還を", ["保証金の変換を", "保証金の返還を"], "長い一セグメント"),
    wide("しけんのかいとうを", "採点者は", "試験の解答を", ["試験の回答を", "試験の解答を", "試験の解凍を"], "長い一セグメント"),
    wide("とりひきさきのたいしょうを", "監査では", "取引先の対象を", ["取引先の対照を", "取引先の対象を", "取引先の対称を"], "長い一セグメント"),
    wide("じんじのいどうを", "社内通知で", "人事の異動を", ["人事の移動を", "人事の異動を", "人事の異同を"], "長い一セグメント"),
    wide("しんせいしょのはっこうを", "窓口では", "申請書の発行を", ["申請書の発効を", "申請書の発行を", "申請書の発光を"], "長い一セグメント"),
]

WIDE_CONTROL = [
    wide("きかくしょのこうせいを", "提出前に", "規格書の校正を", ["規格書の校正を", "規格書の構成を", "規格書の更正を"], "長い一セグメント", True),
    wide("こうじげんばのしょうがいが", "監視中に", "工事現場の障害が", ["工事現場の障害が", "工事現場の傷害が"], "長い一セグメント", True),
    wide("けいやくしょのきかんを", "更新時は", "契約書の期間を", ["契約書の期間を", "契約書の機関を", "契約書の器官を"], "長い一セグメント", True),
    wide("とうあんのかいとうを", "採点では", "答案の解答を", ["答案の解答を", "答案の回答を", "答案の解凍を"], "長い一セグメント", True),
]


def make_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for group, items in (("stress", BASE), ("control", CONTROL_BASE)):
        for index, item in enumerate(items, start=1):
            for mode in ("lexical", "particle"):
                if mode == "lexical":
                    reading = item["reading"]
                    candidates = item["candidates"]
                    expected = item["expected"]
                else:
                    reading = item["reading"] + item["particle"]
                    candidates = [word + item["particle"] for word in item["candidates"]]
                    expected = item["expected"] + item["particle"]
                cases.append({
                    "id": f"{group[0]}{index:03d}-{mode}",
                    "group": group,
                    "mode": mode,
                    "reading": reading,
                    "prefix": item["prefix"],
                    "expected": expected,
                    "candidates": candidates,
                    "category": item["category"],
                })
    for index, item in enumerate(WIDE_STRESS + WIDE_CONTROL, start=1):
        cases.append({
            "id": f"w{index:03d}-wide",
            "group": "control" if item["control"] else "stress",
            "mode": "wide",
            **item,
        })
    return cases


CASES = make_cases()


def _make_request(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "request_id": f"preceding-only-{index:03d}",
        "preceding_text": item["prefix"],
        "following_text": "",
        "read": item["reading"],
        "candidates": [
            {"id": f"c{i}", "text": text, "rank": i}
            for i, text in enumerate(item["candidates"])
        ],
    }


def _make_ranker(path: Path) -> OnnxRuriReranker:
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


def verify_unseen() -> None:
    known = set()
    for path in (ROOT / "integration").glob("*.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    for key in ("positive", "negative", "query", "text", "prompt"):
                        value = row.get(key)
                        if isinstance(value, str):
                            known.add(value)
    old = runpy.run_path(str(ROOT / "scripts" / "create_and_evaluate_holdout_120.py"))["HOLDOUT_TEST_SET"]
    old_prefix = {q["prefix"] for q in old}
    seen = set()
    overlaps = []
    for item in CASES:
        exact = item["prefix"] + item["expected"]
        if exact in seen or exact in known or item["prefix"] in known or item["prefix"] in old_prefix:
            overlaps.append(item["id"])
        seen.add(exact)
    if overlaps:
        raise AssertionError(f"preceding-only holdout leakage: {overlaps}")
    if any(_make_request(item, i)["following_text"] != "" for i, item in enumerate(CASES, 1)):
        raise AssertionError("following_text must be empty for every case")
    print(f"[VERIFICATION] {len(CASES)} preceding-only cases; following_text empty; exact overlaps: 0")


def _pick(ranker: OnnxRuriReranker, request: Dict[str, Any]) -> str:
    response = ranker.rank(request)
    picked = response["candidates"][0]["id"]
    return next(x["text"] for x in request["candidates"] if x["id"] == picked)


def _pick_ensemble(rankers: List[OnnxRuriReranker], request: Dict[str, Any]) -> str:
    explanations = []
    for ranker in rankers:
        ranker.rank(request)
        explanations.append({x["id"]: x for x in ranker.last_explanation["candidates"]})
    ensemble_weights = (
        (0.50, 0.50) if len(str(request.get("read", ""))) <= 3 else (0.25, 0.75)
    )
    scores = {
        cid: sum(
            weight * explanation[cid]["evidence_score"]
            for weight, explanation in zip(ensemble_weights, explanations)
        )
        for cid in explanations[0]
    }
    chosen = max(scores, key=scores.get)
    return next(x["text"] for x in request["candidates"] if x["id"] == chosen)


def evaluate(single_path: Path, ensemble_a: Path, ensemble_b: Path, output: Path) -> Dict[str, Any]:
    verify_unseen()
    single = _make_ranker(single_path)
    ensemble = [_make_ranker(ensemble_a), _make_ranker(ensemble_b)]
    stats: Dict[str, Dict[str, int]] = {}
    details = []
    for index, item in enumerate(CASES, start=1):
        req = _make_request(item, index)
        single_pick = _pick(single, req)
        ensemble_pick = _pick_ensemble(ensemble, req)
        key = f"{item['group']}/{item['mode']}"
        row = stats.setdefault(key, {"total": 0, "single": 0, "ensemble": 0})
        single_ok = single_pick == item["expected"]
        ensemble_ok = ensemble_pick == item["expected"]
        row["total"] += 1
        row["single"] += int(single_ok)
        row["ensemble"] += int(ensemble_ok)
        details.append({
            **item,
            "single": single_pick,
            "ensemble": ensemble_pick,
            "single_ok": single_ok,
            "ensemble_ok": ensemble_ok,
        })
    for row in stats.values():
        row["single_accuracy"] = round(row["single"] / row["total"] * 100.0, 2)
        row["ensemble_accuracy"] = round(row["ensemble"] / row["total"] * 100.0, 2)
    result = {"total": len(CASES), "stats": stats, "details": details}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(CASES), "stats": stats}, ensure_ascii=False, indent=2))
    print("[MISSES]")
    for row in details:
        if not row["single_ok"] or not row["ensemble_ok"]:
            print(json.dumps({k: row[k] for k in ("id", "group", "mode", "category", "prefix", "reading", "expected", "single", "ensemble")}, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-a", type=Path, default=ROOT / "build" / "onnx-model-70m-lora3-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-b", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "preceding_only_range_holdout_evaluation.json")
    args = parser.parse_args()
    evaluate(args.single.resolve(), args.ensemble_a.resolve(), args.ensemble_b.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
