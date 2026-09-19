"""Practical Japanese IME holdout with particles inside and outside a segment.

Each base sentence is evaluated twice:

* ``separate``: the candidate is the lexical word and the particle is in the
  following context (e.g. reading ``はな`` + suffix ``が詰まる``).
* ``attached``: the candidate and reading include the particle (e.g. ``はなが``
  and candidates ``鼻が``/``花が``).

This is intentionally a fresh, leakage-checked benchmark.  It is a diagnostic
benchmark only; its examples are never added to LoRA training data.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker


def b(
    reading: str,
    prefix: str,
    rest: str,
    particle: str,
    expected: str,
    candidates: List[str],
    category: str,
) -> Dict[str, Any]:
    assert expected in candidates and candidates[0] != expected
    return {
        "reading": reading,
        "prefix": prefix,
        "rest": rest,
        "particle": particle,
        "expected": expected,
        "candidates": candidates,
        "category": category,
    }


def control(
    reading: str,
    prefix: str,
    rest: str,
    particle: str,
    expected: str,
    candidates: List[str],
    category: str,
) -> Dict[str, Any]:
    """A natural control where the Mozc first candidate is already correct."""
    assert expected in candidates and candidates[0] == expected
    return {
        "reading": reading,
        "prefix": prefix,
        "rest": rest,
        "particle": particle,
        "expected": expected,
        "candidates": candidates,
        "category": category,
    }


BASE: List[Dict[str, Any]] = [
    # Everyday nouns.
    b("はな", "診察で患者の", "詰まっている。", "が", "鼻", ["花", "鼻"], "日常語"),
    b("はな", "玄関脇の鉢植えの", "元気に咲いた。", "が", "花", ["鼻", "花"], "日常語"),
    b("はし", "昼食では", "使って食べた。", "を", "箸", ["橋", "箸", "端"], "日常語"),
    b("はし", "駅員は新しい", "渡った。", "を", "橋", ["箸", "橋", "端"], "日常語"),
    b("はし", "書類の右", "押印した。", "に", "端", ["橋", "箸", "端"], "日常語"),
    b("あめ", "夕方から", "強く降り始めた。", "が", "雨", ["飴", "雨"], "日常語"),
    b("あめ", "休憩室で", "一粒ずつ配った。", "を", "飴", ["雨", "飴"], "日常語"),
    b("かみ", "美容師が長い", "丁寧に整えた。", "を", "髪", ["紙", "髪", "神"], "日常語"),
    b("かみ", "担当者は", "手順を書き出した。", "に", "紙", ["髪", "紙", "神"], "日常語"),
    b("くも", "庭の植木鉢を", "這っているのを見つけた。", "が", "蜘蛛", ["雲", "蜘蛛"], "日常語"),
    # Verbs with the quotative particle, a common natural segmentation point.
    b("あう", "この寸法なら部品は規格に", "判断できる。", "と", "合う", ["会う", "合う", "遭う"], "異字同訓"),
    b("あう", "駅で昔の同僚に", "声をかける。", "と", "会う", ["合う", "会う", "遭う"], "異字同訓"),
    b("あう", "山道では急な落石に", "大けがをする。", "と", "遭う", ["会う", "遭う", "合う"], "異字同訓"),
    b("あける", "安全確認後に担当者が扉を", "作業を始める。", "と", "開ける", ["明ける", "開ける", "空ける"], "異字同訓"),
    b("あける", "東の空が明るくなり夜が", "静かに朝へ変わる。", "と", "明ける", ["開ける", "明ける", "空ける"], "異字同訓"),
    b("あける", "荷物を全部出して部屋を", "掃除を始めた。", "と", "空ける", ["開ける", "空ける", "明ける"], "異字同訓"),
    b("かえる", "運用開始後に設定を", "不具合が減った。", "と", "変える", ["替える", "変える", "換える", "代える"], "異字同訓"),
    b("かえる", "冬服を新しいサイズに", "動きやすくなった。", "と", "替える", ["変える", "替える", "換える", "代える"], "異字同訓"),
    b("かえる", "窓口で外貨を", "受け取った。", "と", "換える", ["変える", "換える", "替える", "代える"], "異字同訓"),
    b("かえる", "先発投手を", "継投に切り替えた。", "と", "代える", ["変える", "代える", "替える", "換える"], "異字同訓"),
    b("さす", "地図上で目的地を", "確認した。", "と", "指す", ["差す", "指す", "刺す"], "異字同訓"),
    b("さす", "日差しが強く傘を", "歩いた。", "と", "差す", ["指す", "差す", "刺す"], "異字同訓"),
    b("さす", "針先で布を", "縫い合わせる。", "と", "刺す", ["指す", "刺す", "差す"], "異字同訓"),
    # Business and administration.
    b("しよう", "製品マニュアルの", "確認を終えた。", "を", "仕様", ["使用", "仕様", "試用"], "実務"),
    b("しよう", "共有端末の", "許可した。", "を", "使用", ["仕様", "使用", "試用"], "実務"),
    b("ほしょう", "物流遅延で生じた損害の", "申請を受け付けた。", "を", "補償", ["保証", "補償", "保障"], "実務"),
    b("ほしょう", "メーカーによる品質の", "期間を確認した。", "を", "保証", ["補償", "保証", "保障"], "実務"),
    b("ほしょう", "地域の安全の", "制度を整備した。", "を", "保障", ["保証", "保障", "補償"], "実務"),
    b("はっこう", "この規約は来月から", "効力を持つ。", "が", "発効", ["発行", "発効", "発光"], "実務"),
    b("はっこう", "申請者は証明書の", "依頼した。", "を", "発行", ["発効", "発行", "発光"], "実務"),
    b("かいてい", "研究論文の記述の", "行った。", "を", "改訂", ["改定", "改訂", "開廷"], "実務"),
    b("かいてい", "来年度からの公共料金の", "発表した。", "を", "改定", ["改訂", "改定", "開廷"], "実務"),
    b("きてい", "社内服務に関する", "を改めた。", "を", "規程", ["規定", "規程", "既定"], "実務"),
    b("きてい", "契約書における上限額の", "確認した。", "を", "規定", ["規程", "規定", "既定"], "実務"),
    b("けっさい", "部長の", "を経て発注した。", "を", "決裁", ["決済", "決裁", "結済"], "実務"),
    b("けっさい", "オンラインでの代金", "完了した。", "を", "決済", ["決裁", "決済", "結済"], "実務"),
    b("こうせい", "印刷前に原稿の", "を依頼した。", "を", "校正", ["構成", "校正", "更正"], "実務"),
    b("こうせい", "画面のレイアウト", "を見直した。", "を", "構成", ["校正", "構成", "更生"], "実務"),
    b("せいさん", "出張後の交通費", "済ませた。", "を", "精算", ["生産", "精算", "清算"], "実務"),
    b("せいさん", "工場の今月の部品", "計画を見直した。", "を", "生産", ["精算", "生産", "清算"], "実務"),
    b("しょうがい", "システムの", "長時間続いた。", "が", "障害", ["傷害", "障害", "渉外"], "実務"),
    b("しょうがい", "通勤中の事故で", "発生した。", "が", "傷害", ["障害", "傷害", "渉外"], "実務"),
    b("のうき", "納品先と", "確定した。", "を", "納期", ["能率", "納期", "農機"], "実務"),
    # Technical, legal, and research contexts.
    b("いどう", "人事部は来月の配置", "通知した。", "を", "異動", ["移動", "異動", "異同"], "技術・法務"),
    b("いどう", "荷物の", "台車で行う。", "を", "移動", ["異動", "移動", "異同"], "技術・法務"),
    b("かいとう", "受験者は記述式の", "見直した。", "を", "解答", ["回答", "解答", "解凍"], "技術・法務"),
    b("かいとう", "担当者は問い合わせへの", "送った。", "を", "回答", ["解答", "回答", "解凍"], "技術・法務"),
    b("たいしょう", "二つの実験結果の", "実施した。", "を", "対照", ["対象", "対照", "対称"], "技術・法務"),
    b("たいしょう", "監査では取引先", "に含める。", "を", "対象", ["対照", "対象", "対称"], "技術・法務"),
    b("へんかん", "仕様書の別形式への", "実施した。", "を", "変換", ["返還", "変換"], "技術・法務"),
    b("へんかん", "契約終了後の保証金の", "求めた。", "を", "返還", ["変換", "返還"], "技術・法務"),
    b("けいき", "制度変更を", "として研究を始めた。", "に", "契機", ["景気", "契機", "経期"], "技術・法務"),
    b("けいき", "国内の", "緩やかに回復した。", "が", "景気", ["契機", "景気", "敬意"], "技術・法務"),
    b("きかん", "契約の有効", "延長した。", "を", "期間", ["機関", "期間", "器官"], "技術・法務"),
    b("きかん", "患者の", "チューブを入れる。", "を", "気管", ["期間", "気管", "機関"], "技術・法務"),
    b("しこう", "彼の論理的な", "分析する。", "を", "思考", ["試行", "思考", "志向"], "技術・法務"),
]


CONTROL_BASE: List[Dict[str, Any]] = [
    control("はな", "診察前に患者の", "痛みを聞いた。", "の", "鼻", ["鼻", "花"], "日常語"),
    control("はな", "窓辺の", "水をやった。", "に", "花", ["花", "鼻"], "日常語"),
    control("はし", "通勤客は歩道", "渡った。", "を", "橋", ["橋", "箸", "端"], "日常語"),
    control("あめ", "予報どおり午後は", "なった。", "に", "雨", ["雨", "飴"], "日常語"),
    control("かみ", "美容院で", "短く切った。", "を", "髪", ["髪", "紙", "神"], "日常語"),
    control("しよう", "製品の詳細な", "担当者に確認した。", "を", "仕様", ["仕様", "使用", "試用"], "実務"),
    control("しよう", "実験用タブレットの", "申請した。", "を", "使用", ["使用", "仕様", "試用"], "実務"),
    control("ほしょう", "購入品のメーカー", "保管した。", "を", "保証", ["保証", "補償", "保障"], "実務"),
    control("けっさい", "請求書のオンライン", "完了した。", "を", "決済", ["決済", "決裁", "結済"], "実務"),
    control("こうせい", "入稿前に原稿の", "担当者へ送った。", "を", "校正", ["校正", "構成", "更正"], "実務"),
    control("せいさん", "出張後の経費", "週内に済ませた。", "を", "精算", ["精算", "生産", "清算"], "実務"),
    control("いどう", "人事部から社員の", "知らせる通知が来た。", "を", "異動", ["異動", "移動", "異同"], "技術・法務"),
]


def make_cases(items: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    cases = []
    for index, item in enumerate(items, start=1):
        particle = item["particle"]
        for mode in ("separate", "attached"):
            if mode == "separate":
                reading = item["reading"]
                candidates = item["candidates"]
                expected = item["expected"]
                suffix = particle + item["rest"]
            else:
                reading = item["reading"] + particle
                candidates = [word + particle for word in item["candidates"]]
                expected = item["expected"] + particle
                suffix = item["rest"]
            cases.append({
                "id": f"{kind[0]}{index:03d}-{mode}",
                "base_id": index,
                "kind": kind,
                "mode": mode,
                "reading": reading,
                "prefix": item["prefix"],
                "suffix": suffix,
                "expected": expected,
                "candidates": candidates,
                "category": item["category"],
            })
    return cases


CASES = make_cases(BASE, "stress") + make_cases(CONTROL_BASE, "control")


def _make_request(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "request_id": f"particle-holdout-{index:03d}",
        "preceding_text": item["prefix"],
        "following_text": item["suffix"],
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
    old_full = {q["prefix"] + q["expected"] + q["suffix"] for q in old}
    old_prefix = {q["prefix"] for q in old}
    overlaps = []
    fulls: Dict[str, tuple[str, int]] = {}
    for item in CASES:
        full = item["prefix"] + item["expected"] + item["suffix"]
        base_key = (item["kind"], item["base_id"])
        duplicate_pair = full in fulls and fulls[full] == base_key
        duplicate_other = full in fulls and fulls[full] != base_key
        if duplicate_other or full in known or item["prefix"] in known or full in old_full or item["prefix"] in old_prefix:
            overlaps.append(item["id"])
        if not duplicate_pair:
            fulls[full] = base_key
    if overlaps:
        raise AssertionError(f"particle holdout leakage: {overlaps}")
    print(f"[VERIFICATION] {len(CASES)} cases ({len(BASE)} stress + {len(CONTROL_BASE)} control sentences x 2 segmentations); exact overlaps: 0")


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
    ens = [_make_ranker(ensemble_a), _make_ranker(ensemble_b)]
    stats = {
        kind: {mode: {"total": 0, "single": 0, "ensemble": 0} for mode in ("separate", "attached")}
        for kind in ("stress", "control")
    }
    categories: Dict[str, Dict[str, int]] = {}
    details = []
    for index, item in enumerate(CASES, start=1):
        req = _make_request(item, index)
        single_pick = _pick(single, req)
        ensemble_pick = _pick_ensemble(ens, req)
        single_ok = single_pick == item["expected"]
        ensemble_ok = ensemble_pick == item["expected"]
        mode_stats = stats[item["kind"]][item["mode"]]
        mode_stats["total"] += 1
        mode_stats["single"] += int(single_ok)
        mode_stats["ensemble"] += int(ensemble_ok)
        cat = item["category"]
        cstats = categories.setdefault(cat, {"total": 0, "single": 0, "ensemble": 0})
        cstats["total"] += 1
        cstats["single"] += int(single_ok)
        cstats["ensemble"] += int(ensemble_ok)
        details.append({
            **item,
            "context": item["prefix"] + "[" + item["expected"] + "]" + item["suffix"],
            "single": single_pick,
            "ensemble": ensemble_pick,
            "single_ok": single_ok,
            "ensemble_ok": ensemble_ok,
        })
    result = {
        "total": len(CASES),
        "base_sentences": len(BASE),
        "stats": stats,
        "categories": categories,
        "details": details,
    }
    for kind_stats in stats.values():
        for mode_stats in kind_stats.values():
            for key in ("single", "ensemble"):
                mode_stats[key + "_accuracy"] = round(mode_stats[key] / mode_stats["total"] * 100.0, 2)
    control_false_switches = {
        "single": sum(1 for row in details if row["kind"] == "control" and not row["single_ok"]),
        "ensemble": sum(1 for row in details if row["kind"] == "control" and not row["ensemble_ok"]),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["control_false_switches"] = control_false_switches
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(CASES), "stress_sentences": len(BASE), "control_sentences": len(CONTROL_BASE), "stats": stats, "control_false_switches": control_false_switches}, ensure_ascii=False, indent=2))
    print("[MISSES]")
    for row in details:
        if not row["single_ok"] or not row["ensemble_ok"]:
            print(json.dumps({k: row[k] for k in ("id", "mode", "category", "context", "expected", "single", "ensemble")}, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-a", type=Path, default=ROOT / "build" / "onnx-model-70m-lora3-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-b", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "practical_particle_holdout_evaluation.json")
    args = parser.parse_args()
    evaluate(args.single.resolve(), args.ensemble_a.resolve(), args.ensemble_b.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
