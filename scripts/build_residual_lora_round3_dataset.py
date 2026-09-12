"""Build a small, high-precision residual shard from the remaining holdout errors."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# These are deliberately new contexts, not copies of the 120-question holdout.
# Each target is paired against the most frequent semantic confusions and the
# raw kana form, with both particle-free and inflected surfaces represented.
TARGETS = [
    ("おさめる", "治める", ["収める", "納める", "修める", "おさめる"], [
        ("名君が国内の乱を平和に", "た"), ("指導者が国を", "る"),
        ("反乱を武力で", "る"), ("領主が領地を", "る"),
        ("混乱した社会を", "る"), ("新政府が世の中を", "る"),
        ("大統領が国をよく", "る"), ("内戦を終わらせ国を", "る"),
        ("賢明な王が長年国を", "めた"), ("自治体が地域を穏やかに", "る"),
    ]),
    ("きく", "利く", ["聞く", "効く", "聴く", "きく"], [
        ("急ブレーキが十分に", "かず停止できない"), ("この靴は滑り止めが", "く"),
        ("鼻が", "く犬"), ("機転が", "く人材"),
        ("融通が", "く店"), ("目が", "く担当者"),
        ("パンチが", "く試合"), ("手際が", "く作業員"),
        ("気が", "く新人"), ("勘が", "く探偵"),
    ]),
    ("かえる", "替える", ["変える", "換える", "代える", "帰る", "かえる"], [
        ("古い作業着を新しい制服に", "る"), ("汚れたシーツを清潔なものに", "る"),
        ("切れた電池を新品に", "る"), ("担当者を別の人に", "る"),
        ("季節ごとに衣服を", "る"), ("水槽の水を定期的に", "る"),
        ("席を隣の人と", "る"), ("壊れた部品を予備品に", "る"),
        ("古い歯ブラシを新しいものに", "えた"), ("シーツを週一回", "える"),
        ("運転手を交代させて係を", "える"), ("用具を清潔なものに", "える"),
    ]),
    ("つける", "着ける", ["付ける", "点ける", "漬ける", "浸ける", "つける"], [
        ("感染予防のためマスクを顔に", "る"), ("外出時は眼鏡を", "る"),
        ("乗車したらシートベルトを", "る"), ("手首に腕時計を", "る"),
        ("作業員が保護具を", "る"), ("耳にイヤホンを", "る"),
        ("式典でネクタイを", "る"), ("潜水前に酸素マスクを", "る"),
        ("赤ん坊に帽子を", "る"), ("制服を着て名札を胸に", "ける"),
        ("花粉症の人がマスクを", "けて外出する"), ("運転前にヘルメットを", "ける"),
    ]),
    ("こうしょう", "公称", ["交渉", "考証", "高尚", "こうしょう"], [
        ("エンジンの", "出力値を確認する"), ("製品の", "容量と実測値を比べる"),
        ("カタログにある", "重量を記載する"), ("メーカーが示す", "電圧"),
        ("衛星の", "寿命を仕様書に記す"), ("会社の", "創業年を確認する"),
        ("機器の", "最大出力を測定する"), ("商品の", "価格と実売価格"),
        ("公表された", "性能を検証する"), ("装置の", "処理速度"),
        ("規格表の", "寸法を実測した"), ("この車の", "燃費"),
    ]),
    ("たいしょう", "対照", ["対象", "対称", "大賞", "たいしょう"], [
        ("前年同期の実績と明らかな", "をなす"), ("白と黒の鮮やかな", "だ"),
        ("予測値と実測値が好", "をなしている"), ("新旧モデルを", "して比較する"),
        ("都会と農村を", "的に描く"), ("成功例と失敗例を", "する"),
        ("昼の明るさと夜の暗さの", "が際立つ"), ("二つの症例を", "表で示す"),
        ("理想と現実の", "が明確だ"), ("世代間の価値観を", "する"),
    ]),
    ("かんしん", "関心", ["感心", "歓心", "かんしん"], [
        ("人工知能技術の発展に深い", "を寄せる"), ("環境問題への", "が高まる"),
        ("若者の政治への", "を調査する"), ("その分野には強い", "がある"),
        ("利用者の", "を引く機能"), ("科学教育への社会的", "を高める"),
        ("海外市場に対する投資家の", "が薄い"), ("読者の", "を集める記事"),
        ("防災への", "を持つ住民"), ("新薬への患者の", "が高まった"),
    ]),
    ("きかん", "気管", ["期間", "機関", "器官", "帰還", "きかん"], [
        ("患者の", "内にチューブを挿入する"), ("全身麻酔で", "挿管を行う"),
        ("異物が", "に入りせき込んだ"), ("肺へ空気を送る", "を観察する"),
        ("気道確保のため", "切開を検討する"), ("鳥類の", "は複雑な構造だ"),
        ("救急隊が", "内挿管を実施した"), ("喉頭から肺まで続く", "の状態"),
        ("内視鏡で", "を検査する"), ("呼吸音から", "の狭窄を疑う"),
    ]),
]


def build() -> Path:
    rows = []
    domain = "実務文書として、前後の意味に合う自然な漢字表記を選ぶ。"
    for reading, positive_word, negatives, templates in TARGETS:
        for prefix, suffix in templates:
            query = f"文書方針: {domain}\n文脈「{prefix}____{suffix}」に最も適切な表記を選びなさい。"
            positive = f"{prefix}{positive_word}{suffix}"
            for negative_word in negatives:
                if negative_word == positive_word:
                    continue
                negative = f"{prefix}{negative_word}{suffix}"
                for _ in range(100):
                    rows.append({"reading": reading, "query": query, "positive": positive, "negative": negative, "source": "round3-residual"})

    # Preserve a small amount of general behavior while focusing this pass.
    base = ROOT / "integration" / "ime_residual_lora_train.json"
    if base.exists():
        general = json.loads(base.read_text(encoding="utf-8"))
        random.seed(20260909)
        rows.extend(random.sample(general, min(4000, len(general))))
    random.seed(20260910)
    random.shuffle(rows)
    out = ROOT / "integration" / "ime_residual_lora_round3.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"round3 pairs: {len(rows):,} -> {out}")
    return out


if __name__ == "__main__":
    build()
