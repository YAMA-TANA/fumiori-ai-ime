"""High-precision final residual shard for the last strict-holdout confusions."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ("あたたかい", "温かい", ["暖かい", "あたたかい"], [
        ("寒い日に飲む", "ミルクが体を温める"), ("出来たての", "味噌汁をすする"),
        ("手のひらで", "湯飲みを包む"), ("焼きたての", "パンを切る"),
        ("患者に", "お茶を差し出す"), ("鍋から", "スープを取り分ける"),
        ("炊飯器を開けると", "ご飯の香りがした"), ("冷えた体に", "タオルを当てる"),
        ("母の", "手料理を囲む"), ("心のこもった", "歓迎を受ける"),
    ]),
    ("こうしょう", "高尚", ["交渉", "公称", "考証", "こうしょう"], [
        ("俗世を離れた品格ある古典芸能や", "な趣味を嗜む"),
        ("学問的で", "な議論を展開する"), ("精神性の", "な目標を掲げる"),
        ("人格を高める", "な読書"), ("単なる娯楽を超えた", "な作品だ"),
        ("教養に裏打ちされた", "な趣味を持つ"), ("知的で", "な会話を楽しむ"),
        ("倫理的で", "な理想を追求する"), ("芸術性の", "な表現を評価する"),
        ("読者に", "な刺激を与える評論"),
    ]),
    ("たいしょう", "対照", ["対象", "対称", "大賞", "たいしょう"], [
        ("前年同期の実績と明らかな", "をなす"), ("予測値と実測値が好", "をなす"),
        ("白と黒の鮮やかな", "だ"), ("成功例と失敗例を", "する"),
        ("新旧モデルを", "表に並べる"), ("都会と地方を", "的に描く"),
        ("理想と現実の", "が際立つ"), ("二つの症例を", "して説明する"),
        ("昼と夜の明暗が", "的だ"), ("両者の違いを", "させる"),
    ]),
    ("かんしん", "関心", ["感心", "歓心", "かんしん"], [
        ("人工知能技術の発展に深い", "を寄せている"), ("環境問題への", "が高まる"),
        ("若者の政治への", "を調査する"), ("その分野には強い", "がある"),
        ("利用者の", "を引く新機能"), ("科学教育への社会的", "を高める"),
        ("海外市場に対する投資家の", "が薄い"), ("防災への", "を持つ住民"),
        ("新薬への患者の", "が高まった"), ("読者の", "を集める特集記事"),
    ]),
    ("ふっきゅう", "復旧", ["復帰", "復興", "ふっきゅう"], [
        ("サーバー障害からシステムを", "させる"), ("停電後に電力供給を", "する"),
        ("バックアップからデータを", "する"), ("通信回線を速やかに", "する"),
        ("災害で止まった設備を", "させる"), ("ネットワーク機器を再起動して", "する"),
        ("障害対応班がサービスを", "させた"), ("断線した回線を夜通しで", "した"),
        ("工場の生産ラインを", "させる"), ("システムの", "作業が完了した"),
    ]),
]


def build() -> Path:
    rows = []
    domain = "実務文書として前後の意味に合う自然で正確な表記を選ぶ。"
    for reading, positive_word, negatives, templates in TARGETS:
        for prefix, suffix in templates:
            query = f"文書方針: {domain}\n文脈「{prefix}____{suffix}」に最も適切な表記を選びなさい。"
            positive = f"{prefix}{positive_word}{suffix}"
            for negative_word in negatives:
                for _ in range(100):
                    rows.append({"reading": reading, "query": query, "positive": positive, "negative": f"{prefix}{negative_word}{suffix}", "source": "round4-residual"})

    base = ROOT / "integration" / "ime_residual_lora_train.json"
    if base.exists():
        general = json.loads(base.read_text(encoding="utf-8"))
        random.seed(20260911)
        rows.extend(random.sample(general, min(5000, len(general))))
    random.seed(20260911)
    random.shuffle(rows)
    out = ROOT / "integration" / "ime_residual_lora_round4.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"round4 pairs: {len(rows):,} -> {out}")
    return out


if __name__ == "__main__":
    build()
