"""Morphology-focused residual shard for four stubborn surface forms."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ("あたたかい", "温かい", ["暖かい", "あたたかい"], [
        ("寒い冬の夜に飲む", "ココアが格別だ"), ("冷えた朝に口にする", "スープが身に染みる"),
        ("カップから湯気が立つ", "コーヒーを飲む"), ("出来たての", "お茶をゆっくり味わう"),
        ("湯気の立つ", "うどんを食べる"), ("電子レンジで", "弁当を温める"),
        ("手にした", "飲み物で体を温める"), ("鍋の", "料理を家族で囲む"),
        ("焼きたてでまだ", "パンを切り分ける"), ("患者へ", "白湯を差し出す"),
    ]),
    ("かえる", "替える", ["変える", "換える", "代える", "かえる"], [
        ("古い作業着を新しいユニフォームに", "る"), ("汚れたシーツを清潔なものに", "る"),
        ("使い終えた電池を新品に", "る"), ("担当者を別のスタッフに", "る"),
        ("季節に合わせて衣服を", "る"), ("水槽の水を定期的に", "る"),
        ("席を隣の人と", "る"), ("壊れた部品を予備品に", "る"),
        ("古い用具を新しい道具に", "る"), ("当番を次の人に", "る"),
        ("古いパスワードを新しいものに", "る"), ("タオルを清潔なものに", "る"),
    ]),
    ("つける", "着ける", ["付ける", "点ける", "漬ける", "浸ける", "つける"], [
        ("感染予防のため人混みではマスクを顔に", "る"), ("外出前に眼鏡を", "る"),
        ("乗車したらシートベルトを", "る"), ("手首に腕時計を", "る"),
        ("作業員が保護具を", "る"), ("耳にイヤホンを", "る"),
        ("式典でネクタイを", "る"), ("潜水前に酸素マスクを", "る"),
        ("赤ん坊に帽子を", "る"), ("運転前にヘルメットを", "る"),
    ]),
    ("きかん", "気管", ["期間", "機関", "器官", "帰還", "きかん"], [
        ("呼吸困難の患者に対して気管内", "挿管を実施する"), ("救急隊が気道確保のため気管内", "挿管を行う"),
        ("全身麻酔中の患者の気管内", "挿管を確認する"), ("異物が", "に入りせき込んだ"),
        ("肺へ空気を送る", "を内視鏡で調べる"), ("気道確保のため", "切開を検討する"),
        ("鳥類の", "は複雑な構造を持つ"), ("喉から肺まで続く", "の状態"),
        ("手術室で患者の", "を確保する"), ("人工呼吸器と", "チューブを接続する"),
    ]),
]


def build() -> Path:
    rows = []
    domain = "文脈と活用形を重視し、自然で正確な日本語表記を選ぶ。"
    for reading, positive_word, negatives, templates in TARGETS:
        for prefix, suffix in templates:
            query = f"文書方針: {domain}\n文脈「{prefix}____{suffix}」に最も適切な表記を選びなさい。"
            positive = f"{prefix}{positive_word}{suffix}"
            for negative_word in negatives:
                for _ in range(200):
                    rows.append({"reading": reading, "query": query, "positive": positive, "negative": f"{prefix}{negative_word}{suffix}", "source": "round5-residual"})
    # A small replay prevents this final narrow pass from erasing other gains.
    base = ROOT / "integration" / "ime_residual_lora_train.json"
    if base.exists():
        general = json.loads(base.read_text(encoding="utf-8"))
        random.seed(20260912)
        rows.extend(random.sample(general, min(4000, len(general))))
    random.seed(20260912)
    random.shuffle(rows)
    out = ROOT / "integration" / "ime_residual_lora_round5.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"round5 pairs: {len(rows):,} -> {out}")
    return out


if __name__ == "__main__":
    build()
