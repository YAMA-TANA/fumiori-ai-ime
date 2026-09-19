"""Build contrastive training dataset with fine-grained homophone hard negatives for 70M Dual-Encoder."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_residual_lora_dataset import HOMOPHONE_FAMILIES

OUTPUT_PATH = ROOT / "build" / "dual_encoder_dataset.json"

# Fine-grained delicate homophone nuances specifically targeted for Dual-Encoder
TARGETED_DELICATE_HOMOPHONES = [
    # 1. 映す / 写す / 移す
    {
        "reading": "うつす",
        "groups": [
            {
                "word": "映す",
                "negs": ["写す", "移す", "うつす"],
                "contexts": [
                    "澄み切った水面に富士山の美しい姿がくっきりと",
                    "姿見の鏡に自分の全身を",
                    "スクリーンの白い布に映像を鮮明に",
                    "夕日に照らされた影を地面に長く",
                    "湖の静かな水面に周囲の紅葉を鮮やかに",
                    "夜空の花火が川面にきらびやかに",
                    "プロジェクターで壁にスライド資料を",
                    "池の水鏡に庭園の松を",
                ]
            },
            {
                "word": "写す",
                "negs": ["映す", "移す", "うつす"],
                "contexts": [
                    "一眼レフカメラで記念写真を美しく",
                    "黒板に書かれた数式を急いでノートに",
                    "契約書の原本をコピー機で鮮明に",
                    "手本となる絵画を模写して精密に",
                    "古文書の文字を一文字ずつ正確に",
                    "観光地で家族の笑顔をカメラに",
                    "重要書類をカラー複写機で",
                ]
            },
            {
                "word": "移す",
                "negs": ["映す", "写す", "うつす"],
                "contexts": [
                    "荷物を別の部屋の棚へ",
                    "次の議題へと出席者の関心を",
                    "本社機能を地方の拠点へと",
                    "風邪のウイルスを他人に",
                    "容器の中の液体を小瓶に",
                    "視線を窓の外の景色へと",
                    "実行計画を行動へと素早く",
                ]
            }
        ]
    },
    # 2. 侵す / 犯す / 冒す
    {
        "reading": "おかす",
        "groups": [
            {
                "word": "侵す",
                "negs": ["犯す", "冒す", "おかす"],
                "contexts": [
                    "外国の軍用機が我が国の領空を不法に",
                    "基本的人権や個人のプライバシーを不当に",
                    "他国の領土や国境を武力で",
                    "著作権や特許権の知財を違法に",
                    "他人の私有地や領域を許可なく",
                    "他国の主権を露骨に",
                ]
            },
            {
                "word": "犯す",
                "negs": ["侵す", "冒す", "おかす"],
                "contexts": [
                    "重大な過ちや取り返しのつかない失敗を",
                    "法律に違反して重大な罪を",
                    "交通規則に違反する不祥事を",
                    "ルールを破り明らかな反則を",
                    "一度犯してしまった罪を深く悔いる",
                ]
            },
            {
                "word": "冒す",
                "negs": ["侵す", "犯す", "おかす"],
                "contexts": [
                    "嵐の中で命の危険を顧みず危険を",
                    "重い病に身体を",
                    "大きなリスクを覚悟の上で危険を",
                    "猛吹雪の危険を顧みずに山へ挑む",
                ]
            }
        ]
    },
    # 3. 温かい / 暖かい
    {
        "reading": "あたたかい",
        "groups": [
            {
                "word": "温かい",
                "negs": ["暖かい", "あたたかい"],
                "contexts": [
                    "寒い冬の日に飲む熱々の",
                    "淹れたての",
                    "湯気の立つ美味しい",
                    "冷えた体を芯から温める",
                    "家族の心遣いや思いやりがこもった",
                    "母が作ってくれた手料理の",
                    "スープやコーヒーなどの",
                    "真心のこもった親切で",
                ]
            },
            {
                "word": "暖かい",
                "negs": ["温かい", "あたたかい"],
                "contexts": [
                    "春の陽気で日差しがとても",
                    "南国の気候で一年中",
                    "暖房の効いた快適で",
                    "羽毛布団に包まれてぐっすり",
                    "小春日和の穏やかで",
                    "冬とは思えないほどポカポカと",
                    "日当たりの良い南向きの部屋は",
                ]
            }
        ]
    },
    # 4. 空ける / 開ける / 明ける
    {
        "reading": "あける",
        "groups": [
            {
                "word": "空ける",
                "negs": ["開ける", "明ける", "あける"],
                "contexts": [
                    "引っ越しの荷物をすべて搬出し部屋を",
                    "来週の会議のためスケジュールを",
                    "お年寄りに電車の座席を",
                    "荷物を置くためにトランクを",
                    "予定をキャンセルして午後を丸ごと",
                    "部屋を引き渡すために荷物を片付けて",
                ]
            },
            {
                "word": "開ける",
                "negs": ["空ける", "明ける", "あける"],
                "contexts": [
                    "換気のために窓を大きく",
                    "鍵を回して玄関のドアを静かに",
                    "プレゼントの包み紙をワクワクしながら",
                    "缶詰のフタを缶切りで",
                    "新しいノートの最初のページを",
                ]
            },
            {
                "word": "明ける",
                "negs": ["空ける", "開ける", "あける"],
                "contexts": [
                    "東の空が白みようやく長い夜が",
                    "年が",
                    "梅雨が",
                    "連休が",
                ]
            }
        ]
    },
    # 5. 揚げる / 上げる / 挙げる
    {
        "reading": "あげる",
        "groups": [
            {
                "word": "揚げる",
                "negs": ["上げる", "挙げる", "あげる"],
                "contexts": [
                    "祝日の朝に玄関先で国旗を高く",
                    "青空高く風に乗せて凧を",
                    "天ぷらや唐揚げを高温の油でカラッと",
                    "港で船から大きな荷物をクレーンで",
                    "祝砲や花火を夜空高く",
                ]
            },
            {
                "word": "上げる",
                "negs": ["揚げる", "挙げる", "あげる"],
                "contexts": [
                    "質問がある人は右手を高く",
                    "徹底的な業務改善で生産性を",
                    "試験で努力して良い成績を",
                    "物価高騰に伴い製品価格を",
                    "大きな歓声を",
                ]
            },
            {
                "word": "挙げる",
                "negs": ["揚げる", "上げる", "あげる"],
                "contexts": [
                    "わかりやすい具体例をいくつか",
                    "全社一丸となって全力を",
                    "犯行の決定的な証拠を",
                    "春に親族を招いて結婚式を",
                ]
            }
        ]
    },
    # 6. 収める / 納める / 治める / 修める
    {
        "reading": "おさめる",
        "groups": [
            {
                "word": "収める",
                "negs": ["納める", "治める", "修める", "おさめる"],
                "contexts": [
                    "決勝戦で劇的な大勝利を",
                    "厳しい交渉の末に見事な成果を",
                    "アルバムに大切な記念写真を",
                    "本棚に読み終えた書籍をきちんと",
                    "大ヒットとなり記録的な興行収入を",
                ]
            },
            {
                "word": "納める",
                "negs": ["収める", "治める", "修める", "おさめる"],
                "contexts": [
                    "期限内に住民税と所得税を",
                    "期日までに完成品を取引先へ",
                    "会費や授業料を窓口で",
                    "神社に絵馬や初穂料を",
                ]
            },
            {
                "word": "治める",
                "negs": ["収める", "納める", "修める", "おさめる"],
                "contexts": [
                    "名君として平和に国を",
                    "発生した暴動や治安の乱れを",
                    "争いを仲裁してその場を平和に",
                ]
            },
            {
                "word": "修める",
                "negs": ["収める", "納める", "治める", "おさめる"],
                "contexts": [
                    "大学院で高度な学問を",
                    "日頃から身を正して徳を",
                    "武道や芸能の道を究めて技芸を",
                ]
            }
        ]
    }
]


def generate_dual_encoder_dataset() -> list[dict]:
    dataset = []

    # 1. Broad systematic coverage from HOMOPHONE_FAMILIES
    for family in HOMOPHONE_FAMILIES:
        reading = family.get("reading", "")
        groups = family.get("groups", [])
        group_words = [g["word"] for g in groups]
        
        for g in groups:
            word = g["word"]
            negs = list(g.get("negs", []))
            for other_w in group_words:
                if other_w != word and other_w not in negs:
                    negs.append(other_w)
                    
            templates = g.get("templates", [])
            for prefix, suffix in templates:
                ctx_preceding = prefix.strip()
                if ctx_preceding:
                    dataset.append({
                        "context": ctx_preceding,
                        "reading": reading,
                        "positive": word,
                        "negatives": negs[:5],
                    })
                if suffix.strip():
                    ctx_both = f"{prefix}_____{suffix}".strip()
                    dataset.append({
                        "context": ctx_both,
                        "reading": reading,
                        "positive": word,
                        "negatives": negs[:5],
                    })

    # 2. Add fine-grained delicate homophone triplets
    for item in TARGETED_DELICATE_HOMOPHONES:
        reading = item["reading"]
        for g in item["groups"]:
            word = g["word"]
            negs = g["negs"]
            for ctx in g["contexts"]:
                # Repeat targeted samples 3 times to ensure strong gradient on subtle nuances
                for _ in range(3):
                    dataset.append({
                        "context": ctx.strip(),
                        "reading": reading,
                        "positive": word,
                        "negatives": negs,
                    })

    random.seed(42)
    random.shuffle(dataset)
    return dataset


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dual_encoder_dataset()
    print(f"Generated {len(dataset)} Dual-Encoder contrastive samples (including fine-grained targeted hard negatives).")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
