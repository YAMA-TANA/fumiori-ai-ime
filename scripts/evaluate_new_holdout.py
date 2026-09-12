"""Evaluate a fresh, leakage-checked 100-question homophone holdout.

The cases below are new sentences, separate from the original 120-question
holdout and all generated LoRA shards.  Mozc is intentionally placed first
with a plausible wrong candidate so the test measures contextual reranking.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker


def c(reading: str, prefix: str, suffix: str, expected: str, candidates: List[str], category: str) -> Dict[str, Any]:
    assert expected in candidates and candidates[0] != expected
    return {
        "reading": reading,
        "prefix": prefix,
        "suffix": suffix,
        "expected": expected,
        "candidates": candidates,
        "category": category,
    }


NEW_HOLDOUT: List[Dict[str, Any]] = [
    # Basic verbs (25)
    c("あう", "新しい取引先と初めて", "。", "会う", ["合う", "会う", "遭う"], "異字同訓"),
    c("あう", "この部品は規格に", "。", "合う", ["会う", "合う", "遭う"], "異字同訓"),
    c("あう", "旅行中に思わぬ災難に", "。", "遭う", ["会う", "遭う", "合う"], "異字同訓"),
    c("あう", "現場の条件がぴたりと", "。", "合う", ["会う", "合う", "遭う"], "異字同訓"),
    c("あう", "退職した恩師に駅で再び", "。", "会う", ["遭う", "会う", "合う"], "異字同訓"),
    c("あける", "担当者が倉庫のシャッターを", "。", "開ける", ["明ける", "開ける", "空ける"], "異字同訓"),
    c("あける", "長い夜がようやく", "。", "明ける", ["開ける", "明ける", "空ける"], "異字同訓"),
    c("あける", "財布の中身をすべて", "。", "空ける", ["開ける", "空ける", "明ける"], "異字同訓"),
    c("あける", "封筒の封を丁寧に", "。", "開ける", ["明ける", "開ける", "空ける"], "異字同訓"),
    c("あける", "新年が", "と気持ちを新たにした。", "明ける", ["開ける", "明ける", "空ける"], "異字同訓"),
    c("おす", "推薦委員会は彼を後任に", "ことにした。", "推す", ["押す", "推す", "おす"], "異字同訓"),
    c("おす", "画面の送信ボタンを", "と処理が始まる。", "押す", ["推す", "押す", "おす"], "異字同訓"),
    c("おす", "専門家はこの候補を強く", "と述べた。", "推す", ["押す", "推す", "おす"], "異字同訓"),
    c("おす", "非常ベルを", "よう指示した。", "押す", ["推す", "押す", "おす"], "異字同訓"),
    c("おす", "私は若手の彼女を代表に", "。", "推す", ["押す", "推す", "おす"], "異字同訓"),
    c("かえる", "会議の方針を大幅に", "必要がある。", "変える", ["替える", "変える", "換える", "代える"], "異字同訓"),
    c("かえる", "古い作業着を新しい制服に", "ことにした。", "替える", ["変える", "替える", "換える", "代える"], "異字同訓"),
    c("かえる", "銀行で日本円をドルに", "。", "換える", ["変える", "換える", "替える", "代える"], "異字同訓"),
    c("かえる", "主力選手に", "て新人を起用する。", "代える", ["変える", "代える", "替える", "換える"], "異字同訓"),
    c("かえる", "働き方そのものを", "改革する。", "変える", ["替える", "変える", "換える", "代える"], "異字同訓"),
    c("さす", "地図上で目的地を", "。", "指す", ["差す", "指す", "注す", "射す"], "異字同訓"),
    c("さす", "雨が降り始めたので傘を", "。", "差す", ["指す", "差す", "注す", "射す"], "異字同訓"),
    c("さす", "容器へスポイトで薬液を一滴", "。", "注す", ["指す", "注す", "差す", "射す"], "異字同訓"),
    c("さす", "雲の間から朝日が", "。", "射す", ["差す", "射す", "指す", "注す"], "異字同訓"),
    c("さす", "針で布を", "。", "刺す", ["指す", "刺す", "差す", "注す"], "異字同訓"),
    # Temperature and つける (10)
    c("あたたかい", "出来たてのスープは", "。", "温かい", ["暖かい", "温かい", "あたたかい"], "異字同訓"),
    c("あたたかい", "春の南風は", "。", "暖かい", ["温かい", "暖かい", "あたたかい"], "異字同訓"),
    c("あたたかい", "手のひらで", "湯飲みを包んだ。", "温かい", ["暖かい", "温かい", "あたたかい"], "異字同訓"),
    c("あたたかい", "暖房の効いた部屋は", "。", "暖かい", ["温かい", "暖かい", "あたたかい"], "異字同訓"),
    c("つける", "契約書に日付を", "。", "付ける", ["着ける", "付ける", "点ける", "漬ける"], "異字同訓"),
    c("つける", "外出前に腕時計を", "。", "着ける", ["付ける", "着ける", "点ける", "漬ける"], "異字同訓"),
    c("つける", "暗くなったので照明を", "。", "点ける", ["付ける", "点ける", "着ける", "漬ける"], "異字同訓"),
    c("つける", "野菜をぬか床に", "。", "漬ける", ["付ける", "漬ける", "着ける", "点ける"], "異字同訓"),
    c("つける", "運転前にヘルメットを", "。", "着ける", ["付ける", "着ける", "点ける", "漬ける"], "異字同訓"),
    c("つける", "料理に塩味を", "。", "付ける", ["着ける", "付ける", "点ける", "漬ける"], "異字同訓"),
    # Business and administration (19)
    c("しよう", "顧客と画面の", "を確認した。", "仕様", ["使用", "仕様", "試用"], "ビジネス"),
    c("しよう", "実験器具を正しく", "する。", "使用", ["仕様", "使用", "試用"], "ビジネス"),
    c("ほしょう", "事故による損害を", "する制度だ。", "補償", ["保証", "補償", "保障"], "ビジネス"),
    c("ほしょう", "メーカーが品質を", "する。", "保証", ["補償", "保証", "保障"], "ビジネス"),
    c("はっこう", "条約の効力が来月から", "する。", "発効", ["発行", "発効", "発光"], "行政"),
    c("はっこう", "窓口で証明書を", "してもらう。", "発行", ["発効", "発行", "発光"], "行政"),
    c("かいてい", "専門書の記述を最新の内容に", "する。", "改訂", ["改定", "改訂", "開廷"], "学術"),
    c("かいてい", "公共料金を来年度から", "する。", "改定", ["改訂", "改定", "開廷"], "行政"),
    c("きてい", "社内服務に関する", "を整備した。", "規程", ["規定", "規程", "既定"], "ビジネス"),
    c("きてい", "利用", "を読んでから登録する。", "規定", ["規程", "規定", "既定"], "ビジネス"),
    c("けっさい", "役員会の", "を経て契約を締結する。", "決裁", ["決済", "決裁", "結済"], "ビジネス"),
    c("けっさい", "オンラインで代金を", "する。", "決済", ["決裁", "決済", "結済"], "ビジネス"),
    c("こうせい", "校閲者が原稿を", "する。", "校正", ["構成", "校正", "公正"], "学術"),
    c("こうせい", "報告書の章立てと", "を見直す。", "構成", ["校正", "構成", "公正"], "学術"),
    c("せいさん", "出張から戻って交通費を", "する。", "精算", ["生産", "精算", "清算"], "ビジネス"),
    c("せいさん", "工場で部品を大量に", "する。", "生産", ["精算", "生産", "清算"], "ビジネス"),
    c("しょうがい", "システムの", "を取り除く。", "障害", ["傷害", "障害", "生涯"], "IT"),
    c("しょうがい", "事件による", "保険に加入した。", "傷害", ["障害", "傷害", "生涯"], "法律"),
    c("のうき", "取引先に製品の", "を回答する。", "納期", ["農機", "納期", "のうき"], "ビジネス"),
    # Everyday nouns (15)
    c("はな", "犬が", "を地面に近づけた。", "鼻", ["花", "鼻", "はな"], "一般"),
    c("はな", "庭に色とりどりの", "が咲く。", "花", ["鼻", "花", "はな"], "一般"),
    c("はな", "象が長い", "で水を吸う。", "鼻", ["花", "鼻", "はな"], "一般"),
    c("はし", "食卓で", "を使って豆をつまむ。", "箸", ["橋", "箸", "端"], "一般"),
    c("はし", "川に新しい", "を架ける。", "橋", ["箸", "橋", "端"], "一般"),
    c("はし", "紙の", "に名前を書く。", "端", ["箸", "端", "橋"], "一般"),
    c("あめ", "子どもに", "を一粒渡す。", "飴", ["雨", "飴", "あめ"], "一般"),
    c("あめ", "午後から", "が降る予報だ。", "雨", ["飴", "雨", "あめ"], "一般"),
    c("あめ", "傘を持たずに", "に濡れた。", "雨", ["飴", "雨", "あめ"], "一般"),
    c("かみ", "美容師が", "を短く切る。", "髪", ["紙", "髪", "神"], "一般"),
    c("かみ", "神社で", "に祈る。", "神", ["髪", "神", "紙"], "一般"),
    c("かみ", "コピー機に", "を補給する。", "紙", ["髪", "紙", "神"], "一般"),
    c("くも", "軒下で", "が巣を張っている。", "蜘蛛", ["雲", "蜘蛛", "くも"], "一般"),
    c("くも", "青空に白い", "が浮かぶ。", "雲", ["蜘蛛", "雲", "くも"], "一般"),
    c("くも", "夕焼けで", "が赤く染まった。", "雲", ["蜘蛛", "雲", "くも"], "一般"),
    # Technical and legal (20)
    c("いどう", "四月の人事", "で新部署へ移った。", "異動", ["移動", "異動", "異同"], "ビジネス"),
    c("いどう", "荷物を別の部屋へ", "する。", "移動", ["異動", "移動", "異同"], "一般"),
    c("いどう", "契約書の新旧版の", "を一覧にする。", "異同", ["移動", "異同", "異動"], "学術"),
    c("かいとう", "数学の模範", "を配布する。", "解答", ["回答", "解答", "解凍"], "学術"),
    c("かいとう", "顧客の問い合わせに", "する。", "回答", ["解答", "回答", "解凍"], "ビジネス"),
    c("こうしょう", "機器の", "出力をカタログに記す。", "公称", ["交渉", "公称", "高尚"], "IT"),
    c("こうしょう", "取引先と納期を", "する。", "交渉", ["公称", "交渉", "考証"], "ビジネス"),
    c("こうしょう", "教養に裏打ちされた", "な趣味だ。", "高尚", ["交渉", "高尚", "公称"], "一般"),
    c("たいしょう", "新旧製品が明らかな", "をなす。", "対照", ["対象", "対照", "対称"], "学術"),
    c("たいしょう", "今回の調査", "は若手社員だ。", "対象", ["対照", "対象", "対称"], "学術"),
    c("たいしょう", "左右", "な図形を描く。", "対称", ["対象", "対称", "対照"], "学術"),
    c("へんかん", "音声を文字データへ", "する。", "変換", ["返還", "変換", "へんかん"], "IT"),
    c("へんかん", "借りた資料を図書館へ", "する。", "返還", ["変換", "返還", "へんかん"], "行政"),
    c("けいき", "今回の受賞を", "に研究を加速する。", "契機", ["景気", "契機", "けいき"], "ビジネス"),
    c("けいき", "国内の", "が緩やかに回復する。", "景気", ["契機", "景気", "けいき"], "経済"),
    c("きかん", "肺や気管支などの呼吸", "を検査する。", "器官", ["期間", "器官", "機関"], "医療"),
    c("きかん", "契約の有効", "を延長する。", "期間", ["器官", "期間", "機関"], "ビジネス"),
    c("きかん", "患者の", "内にチューブを入れる。", "気管", ["期間", "気管", "器官"], "医療"),
    c("かんしん", "人工知能技術に深い", "を寄せる。", "関心", ["感心", "関心", "歓心"], "IT"),
    c("かんしん", "見事な演奏に皆が", "した。", "感心", ["関心", "感心", "歓心"], "一般"),
    # Other high-value verbs (11)
    c("なおす", "医師が傷を", "。", "治す", ["直す", "治す", "なおす"], "医療"),
    c("なおす", "担当者が誤字を", "。", "直す", ["治す", "直す", "なおす"], "ビジネス"),
    c("おさめる", "新政府が国内の混乱を", "る。", "治める", ["収める", "治める", "納める"], "行政"),
    c("おさめる", "納税者が税金を期限内に", "る。", "納める", ["治める", "納める", "収める"], "行政"),
    c("はかる", "実験で液体の体積を", "。", "量る", ["計る", "量る", "測る"], "学術"),
    c("はかる", "二地点間の距離を", "。", "測る", ["計る", "測る", "量る"], "学術"),
    c("きく", "この靴はグリップが", "。", "利く", ["聞く", "利く", "効く"], "一般"),
    c("きく", "薬がよく", "。", "効く", ["利く", "効く", "聞く"], "医療"),
    c("うつす", "鏡に姿を", "。", "映す", ["移す", "映す", "写す"], "一般"),
    c("うつす", "資料を別の部署へ", "。", "移す", ["映す", "移す", "写す"], "ビジネス"),
    c("あらわす", "研究成果を本に", "。", "著す", ["表す", "著す", "現す"], "学術"),
]


def _make_request(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "request_id": f"new-holdout-{index:03d}",
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


def _assert_unseen() -> None:
    known = set()
    # Check every checked-in training shard, not only the residual LoRA files.
    for path in (ROOT / "integration").glob("*.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("positive", "negative", "query", "text", "prompt"):
                value = row.get(key)
                if isinstance(value, str):
                    known.add(value)

    # Also keep this holdout disjoint from the original strict 120-question
    # benchmark, whose source cases live in a Python module rather than JSON.
    old_holdout = runpy.run_path(str(ROOT / "scripts" / "create_and_evaluate_holdout_120.py"))["HOLDOUT_TEST_SET"]
    old_full = {q["prefix"] + q["expected"] + q["suffix"] for q in old_holdout}
    old_prefix = {q["prefix"] for q in old_holdout}
    overlaps = []
    for i, item in enumerate(NEW_HOLDOUT, start=1):
        full = item["prefix"] + item["expected"] + item["suffix"]
        if full in known or item["prefix"] in known or full in old_full or item["prefix"] in old_prefix:
            overlaps.append(i)
    if overlaps:
        raise AssertionError(f"new holdout leakage at cases: {overlaps}")
    print(f"[VERIFICATION] {len(NEW_HOLDOUT)} new cases; exact overlaps: 0")


def evaluate(single_path: Path, ensemble_a: Path, ensemble_b: Path, output: Path) -> Dict[str, Any]:
    _assert_unseen()
    single = _make_ranker(single_path)
    ens_a = _make_ranker(ensemble_a)
    ens_b = _make_ranker(ensemble_b)
    single_correct = 0
    ensemble_correct = 0
    category: Dict[str, Dict[str, int]] = {}
    details = []
    for i, item in enumerate(NEW_HOLDOUT, start=1):
        cat = item["category"]
        category.setdefault(cat, {"total": 0, "single": 0, "ensemble": 0})
        category[cat]["total"] += 1
        req = _make_request(item, i)
        single_response = single.rank(req)
        single_id = single_response["candidates"][0]["id"]
        single_pick = next(x["text"] for x in req["candidates"] if x["id"] == single_id)
        single_ok = single_pick == item["expected"]
        single_correct += int(single_ok)
        category[cat]["single"] += int(single_ok)

        explanations = []
        for ranker in (ens_a, ens_b):
            ranker.rank(req)
            explanations.append({x["id"]: x for x in ranker.last_explanation["candidates"]})
        ensemble_weights = (0.50, 0.50) if len(item["reading"]) <= 3 else (0.25, 0.75)
        scores = {
            cid: sum(
                weight * explanation[cid]["evidence_score"]
                for weight, explanation in zip(ensemble_weights, explanations)
            )
            for cid in explanations[0]
        }
        ensemble_id = max(scores, key=scores.get)
        ensemble_pick = next(x["text"] for x in req["candidates"] if x["id"] == ensemble_id)
        ensemble_ok = ensemble_pick == item["expected"]
        ensemble_correct += int(ensemble_ok)
        category[cat]["ensemble"] += int(ensemble_ok)
        details.append({
            "id": i,
            "category": cat,
            "reading": item["reading"],
            "context": item["prefix"] + "[" + item["expected"] + "]" + item["suffix"],
            "expected": item["expected"],
            "single": single_pick,
            "ensemble": ensemble_pick,
            "single_ok": single_ok,
            "ensemble_ok": ensemble_ok,
        })
    result = {
        "total": len(NEW_HOLDOUT),
        "single_correct": single_correct,
        "single_accuracy": round(single_correct / len(NEW_HOLDOUT) * 100.0, 2),
        "ensemble_correct": ensemble_correct,
        "ensemble_accuracy": round(ensemble_correct / len(NEW_HOLDOUT) * 100.0, 2),
        "ensemble_weights": "short readings 50/50; other readings 25/75",
        "category": category,
        "details": details,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("total", "single_correct", "single_accuracy", "ensemble_correct", "ensemble_accuracy")}, ensure_ascii=False, indent=2))
    for row in details:
        if not row["ensemble_ok"] or not row["single_ok"]:
            print(row["id"], row["context"], "single=", row["single"], "ensemble=", row["ensemble"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-a", type=Path, default=ROOT / "build" / "onnx-model-70m-lora3-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--ensemble-b", type=Path, default=ROOT / "build" / "onnx-model-70m-lora4-20260909" / "ruri-ime-int8.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "new_holdout_100_evaluation.json")
    args = parser.parse_args()
    evaluate(args.single.resolve(), args.ensemble_a.resolve(), args.ensemble_b.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
