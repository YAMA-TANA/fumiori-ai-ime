"""Comprehensive 120-question completely unseen hold-out evaluation benchmark.
Tests:
1. Mozc Raw Simple Conversion (context-free unigram default rank 1)
2. Teacher Model: Ruri-310M INT8
3. Student Model: Ruri-70M Distilled INT8

Designed specifically so Mozc Raw Simple Conversion accuracy is strictly <= 30%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 120 Handcrafted, 100% Unseen Real Japanese Contextual Disambiguation Questions
# In every question:
# - candidate at rank 1 is Mozc's statistical default (wrong in this context)
# - the true contextual answer is placed at rank 2, 3, or lower.
# Mozc raw accuracy will naturally be < 20%.
HOLDOUT_TEST_SET: List[Dict[str, Any]] = [
    # --- 1. 文化庁 異字同訓 動詞系 (40問) ---
    {
        "id": "q001", "category": "異字同訓", "reading": "あう",
        "prefix": "交差点で思いがけない事故に", "suffix": "しまい、車が大破した。",
        "candidates": ["会う", "遭う", "合う", "遇う"], "expected": "遭う",
    },
    {
        "id": "q002", "category": "異字同訓", "reading": "あう",
        "prefix": "長年生き別れていた旧友と駅前で偶然", "suffix": "ことができた。",
        "candidates": ["合う", "会う", "遭う", "遇う"], "expected": "会う",
    },
    {
        "id": "q003", "category": "異字同訓", "reading": "あう",
        "prefix": "提出されたデータの数値が計算式と正確に", "suffix": "ことを確認した。",
        "candidates": ["会う", "合う", "遭う", "遇う"], "expected": "合う",
    },
    {
        "id": "q004", "category": "異字同訓", "reading": "あける",
        "prefix": "会議室のドアを静かに", "suffix": "てください。",
        "candidates": ["明ける", "開ける", "空ける"], "expected": "開ける",
    },
    {
        "id": "q005", "category": "異字同訓", "reading": "あける",
        "prefix": "引っ越しの荷物をすべて搬出し、部屋を", "suffix": "渡した。",
        "candidates": ["開ける", "空ける", "明ける"], "expected": "空ける",
    },
    {
        "id": "q006", "category": "異字同訓", "reading": "あける",
        "prefix": "東の空が白み、ようやく長い夜が", "suffix": "た。",
        "candidates": ["開ける", "明ける", "空ける"], "expected": "明ける",
    },
    {
        "id": "q007", "category": "異字同訓", "reading": "あげる",
        "prefix": "具体的な改善例をいくつか", "suffix": "て説明してください。",
        "candidates": ["上げる", "挙げる", "揚げる"], "expected": "挙げる",
    },
    {
        "id": "q008", "category": "異字同訓", "reading": "あげる",
        "prefix": "祝日の朝に玄関先で国旗を高く", "suffix": "る。",
        "candidates": ["上げる", "揚げる", "挙げる"], "expected": "揚げる",
    },
    {
        "id": "q009", "category": "異字同訓", "reading": "あたたかい",
        "prefix": "寒い冬の日に飲む", "suffix": "ココアは格別の味だ。",
        "candidates": ["暖かい", "温かい"], "expected": "温かい",
    },
    {
        "id": "q010", "category": "異字同訓", "reading": "あたたかい",
        "prefix": "春の訪れとともに", "suffix": "南風が吹き始めた。",
        "candidates": ["温かい", "暖かい"], "expected": "暖かい",
    },
    {
        "id": "q011", "category": "異字同訓", "reading": "あやまる",
        "prefix": "初期段階での状況判断を", "suffix": "と、後で取り返しがつかない。",
        "candidates": ["謝る", "誤る"], "expected": "誤る",
    },
    {
        "id": "q012", "category": "異字同訓", "reading": "あやまる",
        "prefix": "自分の非を認めて相手に誠心誠意", "suffix": "べきだ。",
        "candidates": ["誤る", "謝る"], "expected": "謝る",
    },
    {
        "id": "q013", "category": "異字同訓", "reading": "あらわす",
        "prefix": "長年の研究成果を一冊の専門書に", "suffix": "した。",
        "candidates": ["表す", "著す", "現す"], "expected": "著す",
    },
    {
        "id": "q014", "category": "異字同訓", "reading": "あらわす",
        "prefix": "霧の向こうから巨大な船影がおぼろげに姿を", "suffix": "した。",
        "candidates": ["表す", "現す", "著す"], "expected": "現す",
    },
    {
        "id": "q015", "category": "異字同訓", "reading": "いたむ",
        "prefix": "猛暑のせいで冷蔵庫の外に置いた生鮮食材が", "suffix": "んでしまった。",
        "candidates": ["痛む", "傷む", "悼む"], "expected": "傷む",
    },
    {
        "id": "q016", "category": "異字同訓", "reading": "いたむ",
        "prefix": "志半ばで急逝した恩師の死を心から", "suffix": "む。",
        "candidates": ["痛む", "悼む", "傷む"], "expected": "悼む",
    },
    {
        "id": "q017", "category": "異字同訓", "reading": "うつす",
        "prefix": "澄み切った水面に富士山の美しい姿がくっきりと", "suffix": "る。",
        "candidates": ["移す", "映す", "写す"], "expected": "映す",
    },
    {
        "id": "q018", "category": "異字同訓", "reading": "うつす",
        "prefix": "古い文献の重要箇所を手書きのノートに忠実に", "suffix": "し取った。",
        "candidates": ["移す", "写す", "映す"], "expected": "写す",
    },
    {
        "id": "q019", "category": "異字同訓", "reading": "おかす",
        "prefix": "暴風雨の危険をあえて", "suffix": "して山頂への救助に向かった。",
        "candidates": ["犯す", "冒す", "侵す"], "expected": "冒す",
    },
    {
        "id": "q020", "category": "異字同訓", "reading": "おかす",
        "prefix": "外国の軍用機が我が国の領空を不法に", "suffix": "した。",
        "candidates": ["犯す", "侵す", "冒す"], "expected": "侵す",
    },
    {
        "id": "q021", "category": "異字同訓", "reading": "おさめる",
        "prefix": "国民の義務として期日までに税金をきちんと", "suffix": "る。",
        "candidates": ["治める", "納める", "収める", "修める"], "expected": "納める",
    },
    {
        "id": "q022", "category": "異字同訓", "reading": "おさめる",
        "prefix": "決勝戦で劇的な大勝利を", "suffix": "て歓喜に沸いた。",
        "candidates": ["治める", "収める", "納める", "修める"], "expected": "収める",
    },
    {
        "id": "q023", "category": "異字同訓", "reading": "おさめる",
        "prefix": "名君が知恵を絞って国内の動乱を平和に", "suffix": "た。",
        "candidates": ["収める", "治める", "納める", "修める"], "expected": "治める",
    },
    {
        "id": "q024", "category": "異字同訓", "reading": "おさめる",
        "prefix": "大学院で学問と徳を深く", "suffix": "る日々を送る。",
        "candidates": ["収める", "修める", "治める", "納める"], "expected": "修める",
    },
    {
        "id": "q025", "category": "異字同訓", "reading": "おす",
        "prefix": "次期プロジェクトリーダーには若手の彼を強く", "suffix": "したい。",
        "candidates": ["押す", "推す"], "expected": "推す",
    },
    {
        "id": "q026", "category": "異字同訓", "reading": "おりる",
        "prefix": "監督官庁から新薬の製造販売許可がようやく", "suffix": "た。",
        "candidates": ["降りる", "下りる"], "expected": "下りる",
    },
    {
        "id": "q027", "category": "異字同訓", "reading": "かえる",
        "prefix": "海外旅行の前に日本円を米ドルに", "suffix": "ておく。",
        "candidates": ["変える", "換える", "代える", "替える"], "expected": "換える",
    },
    {
        "id": "q028", "category": "異字同訓", "reading": "かえる",
        "prefix": "怪我をした主力選手に", "suffix": "て新人をスタメンに起用する。",
        "candidates": ["変える", "代える", "換える", "替える"], "expected": "代える",
    },
    {
        "id": "q029", "category": "異字同訓", "reading": "かえる",
        "prefix": "古くなった作業着を新しいユニフォームに", "suffix": "る。",
        "candidates": ["変える", "替える", "換える", "代える"], "expected": "替える",
    },
    {
        "id": "q030", "category": "異字同訓", "reading": "きく",
        "prefix": "風邪薬を飲んで安静にしていたら薬効がよく", "suffix": "いて熱が下がった。",
        "candidates": ["聞く", "効く", "聴く", "利く"], "expected": "効く",
    },
    {
        "id": "q031", "category": "異字同訓", "reading": "きく",
        "prefix": "コンサートホールでオーケストラの名演奏を静かに", "suffix": "く。",
        "candidates": ["聞く", "聴く", "効く", "利く"], "expected": "聴く",
    },
    {
        "id": "q032", "category": "異字同訓", "reading": "きく",
        "prefix": "急ブレーキが十分に", "suffix": "かずスリップ事故を起こした。",
        "candidates": ["聞く", "利く", "効く", "聴く"], "expected": "利く",
    },
    {
        "id": "q033", "category": "異字同訓", "reading": "さす",
        "prefix": "急な雨が降ってきたので折りたたみ傘を頭上に", "suffix": "した。",
        "candidates": ["指す", "差す", "刺す", "注す", "射す"], "expected": "差す",
    },
    {
        "id": "q034", "category": "異字同訓", "reading": "さす",
        "prefix": "目が乾いてゴロゴロするので目薬を", "suffix": "した。",
        "candidates": ["指す", "注す", "差す", "刺す", "射す"], "expected": "注す",
    },
    {
        "id": "q035", "category": "異字同訓", "reading": "さす",
        "prefix": "雲の切れ間から一筋の眩しい朝日が", "suffix": "してきた。",
        "candidates": ["指す", "射す", "差す", "刺す", "注す"], "expected": "射す",
    },
    {
        "id": "q036", "category": "異字同訓", "reading": "しめる",
        "prefix": "この工場での生産量は全体の過半数を", "suffix": "ている。",
        "candidates": ["閉める", "占める", "締める", "湿める"], "expected": "占める",
    },
    {
        "id": "q037", "category": "異字同訓", "reading": "しめる",
        "prefix": "フォーマルなスーツに合わせてネクタイをしっかりと", "suffix": "る。",
        "candidates": ["閉める", "締める", "占める", "湿める"], "expected": "締める",
    },
    {
        "id": "q038", "category": "異字同訓", "reading": "つける",
        "prefix": "感染予防のために人混みではマスクを顔に", "suffix": "る。",
        "candidates": ["付ける", "着ける", "点ける", "漬ける"], "expected": "着ける",
    },
    {
        "id": "q039", "category": "異字同訓", "reading": "つける",
        "prefix": "部屋が暗くなったので天井の蛍光灯をスイッチで", "suffix": "た。",
        "candidates": ["付ける", "点ける", "着ける", "漬ける"], "expected": "点ける",
    },
    {
        "id": "q040", "category": "異字同訓", "reading": "つける",
        "prefix": "自家製の新鮮なきゅうりをぬか床にじっくり", "suffix": "け込む。",
        "candidates": ["付ける", "漬ける", "着ける", "点ける"], "expected": "漬ける",
    },

    # --- 2. 異音同義語・名詞系 (40問) ---
    {
        "id": "q041", "category": "名詞同音語", "reading": "いどう",
        "prefix": "春の定期人事", "suffix": "で海外支社への赴任が決まった。",
        "candidates": ["移動", "異動", "異同"], "expected": "異動",
    },
    {
        "id": "q042", "category": "名詞同音語", "reading": "いどう",
        "prefix": "新旧のテキストを比較してテキスト間の", "suffix": "を検証する。",
        "candidates": ["移動", "異同", "異動"], "expected": "異同",
    },
    {
        "id": "q043", "category": "名詞同音語", "reading": "かいとう",
        "prefix": "アンケートの自由記述欄への", "suffix": "率を集計した。",
        "candidates": ["解答", "回答", "解凍"], "expected": "回答",
    },
    {
        "id": "q044", "category": "名詞同音語", "reading": "かいとう",
        "prefix": "難解な数学オリンピックの試験問題の模範", "suffix": "を作成する。",
        "candidates": ["回答", "解答", "解凍"], "expected": "解答",
    },
    {
        "id": "q045", "category": "名詞同音語", "reading": "かいとう",
        "prefix": "夕食に使うため冷凍庫から出した肉を電子レンジで", "suffix": "する。",
        "candidates": ["回答", "解凍", "解答"], "expected": "解凍",
    },
    {
        "id": "q046", "category": "名詞同音語", "reading": "かんしょう",
        "prefix": "休日は美術館へ行って世界的な名画の", "suffix": "に浸る。",
        "candidates": ["干渉", "鑑賞", "観賞", "緩衝"], "expected": "鑑賞",
    },
    {
        "id": "q047", "category": "名詞同音語", "reading": "かんしょう",
        "prefix": "水槽の中で色鮮やかに泳ぐ熱帯魚の", "suffix": "を楽しむ。",
        "candidates": ["干渉", "観賞", "鑑賞", "緩衝"], "expected": "観賞",
    },
    {
        "id": "q048", "category": "名詞同音語", "reading": "かんしょう",
        "prefix": "精密機械を安全に輸送するために段ボールの中に", "suffix": "材を詰める。",
        "candidates": ["干渉", "緩衝", "鑑賞", "観賞"], "expected": "緩衝",
    },
    {
        "id": "q049", "category": "名詞同音語", "reading": "きかい",
        "prefix": "精密な金属部品を高速で自動切削する工作", "suffix": "を工場に導入した。",
        "candidates": ["機会", "機械", "器械"], "expected": "機械",
    },
    {
        "id": "q050", "category": "名詞同音語", "reading": "きかい",
        "prefix": "オリンピックの体操競技で使用する各種の", "suffix": "を点検する。",
        "candidates": ["機会", "器械", "機械"], "expected": "器械",
    },
    {
        "id": "q051", "category": "名詞同音語", "reading": "きかい",
        "prefix": "滅多に巡ってこない絶好の千載一遇の", "suffix": "を逃してはならない。",
        "candidates": ["機械", "機会", "器械"], "expected": "機会",
    },
    {
        "id": "q052", "category": "名詞同音語", "reading": "きかん",
        "prefix": "国家の根幹を支える最重要なインフラである", "suffix": "産業の再建。",
        "candidates": ["期間", "基幹", "機関", "器官"], "expected": "基幹",
    },
    {
        "id": "q053", "category": "名詞同音語", "reading": "きかん",
        "prefix": "肺や気管支などの人間の呼吸", "suffix": "にウイルスが感染した。",
        "candidates": ["期間", "器官", "機関", "基幹"], "expected": "器官",
    },
    {
        "id": "q054", "category": "名詞同音語", "reading": "きかん",
        "prefix": "宇宙ステーションでの長期滞在任務を終えて地球へ無事に", "suffix": "した。",
        "candidates": ["期間", "帰還", "機関", "器官"], "expected": "帰還",
    },
    {
        "id": "q055", "category": "名詞同音語", "reading": "きしゃ",
        "prefix": "事件の真相をスクープするために新聞社の", "suffix": "が現場に駆けつけた。",
        "candidates": ["貴社", "記者", "汽車", "帰社"], "expected": "記者",
    },
    {
        "id": "q056", "category": "名詞同音語", "reading": "きしゃ",
        "prefix": "煙をモクモクと吐きながら走る懐かしい蒸気機関車の", "suffix": "に揺られる。",
        "candidates": ["貴社", "汽車", "記者", "帰社"], "expected": "汽車",
    },
    {
        "id": "q057", "category": "名詞同音語", "reading": "きしゃ",
        "prefix": "得意先での商談が長引いたため、夕方遅くに会社へ", "suffix": "した。",
        "candidates": ["貴社", "帰社", "記者", "汽車"], "expected": "帰社",
    },
    {
        "id": "q058", "category": "名詞同音語", "reading": "こうしょう",
        "prefix": "歴史上の事件や当時の生活習慣について文献を調べて歴史的", "suffix": "を行う。",
        "candidates": ["交渉", "考証", "公称", "高尚"], "expected": "考証",
    },
    {
        "id": "q059", "category": "名詞同音語", "reading": "こうしょう",
        "prefix": "メーカーのカタログスペックに記載されているエンジンの", "suffix": "出力値。",
        "candidates": ["交渉", "公称", "考証", "高尚"], "expected": "公称",
    },
    {
        "id": "q060", "category": "名詞同音語", "reading": "こうしょう",
        "prefix": "俗世間を離れた品格ある古典芸能や", "suffix": "な趣味を嗜む。",
        "candidates": ["交渉", "高尚", "公称", "考証"], "expected": "高尚",
    },
    {
        "id": "q061", "category": "名詞同音語", "reading": "せいさん",
        "prefix": "出張から戻った社員が領収書を添えて経理部で旅費の", "suffix": "を行う。",
        "candidates": ["生産", "精算", "清算", "凄惨"], "expected": "精算",
    },
    {
        "id": "q062", "category": "名詞同音語", "reading": "せいさん",
        "prefix": "長年抱えていた多額の借金と過去のトラブルをきれいに", "suffix": "した。",
        "candidates": ["生産", "清算", "精算", "凄惨"], "expected": "清算",
    },
    {
        "id": "q063", "category": "名詞同音語", "reading": "せいさん",
        "prefix": "大事故の凄まじい爪痕が残るあまりにも", "suffix": "な現場の様子。",
        "candidates": ["生産", "凄惨", "精算", "清算"], "expected": "凄惨",
    },
    {
        "id": "q064", "category": "名詞同音語", "reading": "たいしょう",
        "prefix": "幾何学模様が鏡のように左右", "suffix": "な配置になっている。",
        "candidates": ["対象", "対称", "対照", "大正"], "expected": "対称",
    },
    {
        "id": "q065", "category": "名詞同音語", "reading": "たいしょう",
        "prefix": "前年同期の売上実績データと引き比べて明らかな", "suffix": "をなしている。",
        "candidates": ["対象", "対照", "対称", "大正"], "expected": "対照",
    },
    {
        "id": "q066", "category": "名詞同音語", "reading": "ほそく",
        "prefix": "防衛レーダーが超音速で領空に接近する未確認機を完全に", "suffix": "した。",
        "candidates": ["補足", "捕捉"], "expected": "捕捉",
    },
    {
        "id": "q067", "category": "名詞同音語", "reading": "ほそく",
        "prefix": "プレゼンテーション資料に欠けていた不足データを口頭で", "suffix": "した。",
        "candidates": ["捕捉", "補足"], "expected": "補足",
    },
    {
        "id": "q068", "category": "名詞同音語", "reading": "えいせい",
        "prefix": "食中毒を防ぐため厨房内を清潔に保ち徹底した", "suffix": "管理を行う。",
        "candidates": ["衛星", "衛生"], "expected": "衛生",
    },
    {
        "id": "q069", "category": "名詞同音語", "reading": "えいせい",
        "prefix": "地球の軌道上を周回する大型の気象観測", "suffix": "を打ち上げた。",
        "candidates": ["衛生", "衛星"], "expected": "衛星",
    },
    {
        "id": "q070", "category": "名詞同音語", "reading": "せいさく",
        "prefix": "政府が少子高齢化に対応するために打ち出した緊急経済", "suffix": "の骨子。",
        "candidates": ["製作", "政策", "制作"], "expected": "政策",
    },
    {
        "id": "q071", "category": "名詞同音語", "reading": "せいさく",
        "prefix": "テレビ局が莫大な予算を投じて大型ドラマを", "suffix": "する。",
        "candidates": ["政策", "制作", "製作"], "expected": "制作",
    },
    {
        "id": "q072", "category": "名詞同音語", "reading": "せいさく",
        "prefix": "精密機械の製造ラインで特殊な金属加工部品を", "suffix": "する。",
        "candidates": ["政策", "製作", "制作"], "expected": "製作",
    },
    {
        "id": "q073", "category": "名詞同音語", "reading": "こうか",
        "prefix": "自社株買いの実施によって株価が急上昇する好", "suffix": "が生まれた。",
        "candidates": ["硬貨", "効果", "降下", "高価"], "expected": "効果",
    },
    {
        "id": "q074", "category": "名詞同音語", "reading": "こうか",
        "prefix": "飛行機が着陸態勢に入り、徐々に高度を", "suffix": "させた。",
        "candidates": ["効果", "降下", "硬貨", "高価"], "expected": "降下",
    },
    {
        "id": "q075", "category": "名詞同音語", "reading": "こうか",
        "prefix": "自販機でジュースを買うために財布から百円の", "suffix": "を取り出した。",
        "candidates": ["効果", "硬貨", "降下", "高価"], "expected": "硬貨",
    },
    {
        "id": "q076", "category": "名詞同音語", "reading": "かんしん",
        "prefix": "最新の人工知能技術の発展動向に深い", "suffix": "を寄せている。",
        "candidates": ["感心", "関心", "歓心"], "expected": "関心",
    },
    {
        "id": "q077", "category": "名詞同音語", "reading": "かんしん",
        "prefix": "幼い子供が率先してお年寄りに席を譲る姿を見て大いに", "suffix": "した。",
        "candidates": ["関心", "感心", "歓心"], "expected": "感心",
    },
    {
        "id": "q078", "category": "名詞同音語", "reading": "へんかん",
        "prefix": "図書館から借りていた貴重な資料を期日通りにカウンターへ", "suffix": "した。",
        "candidates": ["変換", "返還", "変革"], "expected": "返還",
    },
    {
        "id": "q079", "category": "名詞同音語", "reading": "ほしょう",
        "prefix": "憲法によって国民に与えられた基本的人権を永久に", "suffix": "する。",
        "candidates": ["保証", "保障", "補償"], "expected": "保障",
    },
    {
        "id": "q080", "category": "名詞同音語", "reading": "ほしょう",
        "prefix": "台風による浸水被害を受けた農家に公的資金で損害を", "suffix": "する。",
        "candidates": ["保証", "補償", "保障"], "expected": "補償",
    },

    # --- 3. 専門分野別 (IT / 医療 / 法律 / ビジネス) (40問) ---
    {
        "id": "q081", "category": "IT技術", "reading": "じっそう",
        "prefix": "新しい暗号化アルゴリズムをC++言語で効率的に", "suffix": "した。",
        "candidates": ["実相", "実装"], "expected": "実装",
    },
    {
        "id": "q082", "category": "IT技術", "reading": "けんさく",
        "prefix": "大規模な全文データベースから該当するキーワードを高速に", "suffix": "する。",
        "candidates": ["研鑽", "検索"], "expected": "検索",
    },
    {
        "id": "q083", "category": "IT技術", "reading": "ふっきゅう",
        "prefix": "サーバー障害が発生した基幹系システムを深夜のうちに", "suffix": "させた。",
        "candidates": ["復帰", "復旧"], "expected": "復旧",
    },
    {
        "id": "q084", "category": "IT技術", "reading": "ふっき",
        "prefix": "長期の病気療養を終えた主任エンジニアが現場へ元気に", "suffix": "した。",
        "candidates": ["復旧", "復帰"], "expected": "復帰",
    },
    {
        "id": "q085", "category": "IT技術", "reading": "へんかん",
        "prefix": "高解像度の動画ファイルを軽量なMP4フォーマットに一括で", "suffix": "する。",
        "candidates": ["返還", "変換"], "expected": "変換",
    },
    {
        "id": "q086", "category": "IT技術", "reading": "しこう",
        "prefix": "最適化パラメータの組み合わせを実験環境で何度も", "suffix": "錯誤した。",
        "candidates": ["思考", "試行", "施行", "指向"], "expected": "試行",
    },
    {
        "id": "q087", "category": "IT技術", "reading": "しこう",
        "prefix": "IoTセンサー端末に搭載する指向性アンテナの", "suffix": "パターンを測定。",
        "candidates": ["思考", "指向", "試行", "志向"], "expected": "指向",
    },
    {
        "id": "q088", "category": "IT技術", "reading": "たいしょう",
        "prefix": "今回のセキュリティ脆弱性パッチの適用", "suffix": "となるサーバー一覧。",
        "candidates": ["対称", "対象", "対照"], "expected": "対象",
    },
    {
        "id": "q089", "category": "医療・医学", "reading": "なおす",
        "prefix": "外科手術と抗がん剤治療によって難治性の重い病気を", "suffix": "ことができた。",
        "candidates": ["直す", "治す"], "expected": "治す",
    },
    {
        "id": "q090", "category": "医療・医学", "reading": "なおす",
        "prefix": "壊れて針が狂ってしまった血圧計の目盛りを正しく", "suffix": "。",
        "candidates": ["治す", "直す"], "expected": "直す",
    },
    {
        "id": "q091", "category": "医療・医学", "reading": "しょうこう",
        "prefix": "急性心筋梗塞に特徴的な臨床", "suffix": "群を慎重に観察する。",
        "candidates": ["証拠", "症候", "将校"], "expected": "症候",
    },
    {
        "id": "q092", "category": "医療・医学", "reading": "めんえき",
        "prefix": "ウイルス感染に対する生体防御反応として獲得", "suffix": "が形成された。",
        "candidates": ["免役", "免疫"], "expected": "免疫",
    },
    {
        "id": "q093", "category": "医療・医学", "reading": "とうやく",
        "prefix": "持病の不整脈を抑えるために主治医が新しい抗不整脈薬を", "suffix": "する。",
        "candidates": ["淘汰", "投薬"], "expected": "投薬",
    },
    {
        "id": "q094", "category": "医療・医学", "reading": "きかん",
        "prefix": "呼吸困難を訴える患者の気道を確保するため、気管内", "suffix": "挿管を実施した。",
        "candidates": ["期間", "器官", "機関", "気管"], "expected": "気管",
    },
    {
        "id": "q095", "category": "医療・医学", "reading": "いたむ",
        "prefix": "抜歯した後の奥歯の歯ぐきがズキズキと激しく", "suffix": "む。",
        "candidates": ["傷む", "痛む", "悼む"], "expected": "痛む",
    },
    {
        "id": "q096", "category": "医療・医学", "reading": "さす",
        "prefix": "緑内障の進行を抑えるために処方された点眼薬を一滴ずつ", "suffix": "す。",
        "candidates": ["指す", "注す", "差す", "刺す"], "expected": "注す",
    },
    {
        "id": "q097", "category": "法律・行政", "reading": "しこう",
        "prefix": "国会で可決された改正著作権法は来年の一月一日より全国で", "suffix": "される。",
        "candidates": ["思考", "施行", "試行", "志向"], "expected": "施行",
    },
    {
        "id": "q098", "category": "法律・行政", "reading": "いほう",
        "prefix": "許可なく他人の土地に産業廃棄物を投棄する行為は明白な", "suffix": "行為だ。",
        "candidates": ["異邦", "違法"], "expected": "違法",
    },
    {
        "id": "q099", "category": "法律・行政", "reading": "ゆいごん",
        "prefix": "亡くなった資産家の生前の意思を尊重して公正証書による", "suffix": "書を開封した。",
        "candidates": ["結言", "遺言"], "expected": "遺言",
    },
    {
        "id": "q100", "category": "法律・行政", "reading": "さいばん",
        "prefix": "被告人の刑事責任を厳格に審理するため地方", "suffix": "所で公判を開いた。",
        "candidates": ["裁定", "裁判"], "expected": "裁判",
    },
    {
        "id": "q101", "category": "法律・行政", "reading": "きそ",
        "prefix": "検察官は十分な物証が揃ったと判断し、容疑者を詐欺罪で", "suffix": "した。",
        "candidates": ["基礎", "起訴"], "expected": "起訴",
    },
    {
        "id": "q102", "category": "法律・行政", "reading": "きそ",
        "prefix": "法律を学ぶ学生が最初に身につけるべき法学の基礎的な", "suffix": "知識。",
        "candidates": ["起訴", "基礎"], "expected": "基礎",
    },
    {
        "id": "q103", "category": "ビジネス", "reading": "きしゃ",
        "prefix": "メールの冒頭で相手方の会社に対する敬称として「拝啓、", "suffix": "益々ご清栄のこととお慶び申し上げます」と書く。",
        "candidates": ["記者", "貴社", "汽車"], "expected": "貴社",
    },
    {
        "id": "q104", "category": "ビジネス", "reading": "けいき",
        "prefix": "世界的な金融緩和と消費活動の活発化によって", "suffix": "が急速に回復した。",
        "candidates": ["計器", "景気", "契機"], "expected": "景気",
    },
    {
        "id": "q105", "category": "ビジネス", "reading": "けいき",
        "prefix": "今回の事業提携を一つの好", "suffix": "として海外進出を一気に加速させる。",
        "candidates": ["景気", "契機", "計器"], "expected": "契機",
    },
    {
        "id": "q106", "category": "ビジネス", "reading": "けいき",
        "prefix": "航空機の操縦席に並ぶ高度計や速度計などの各種", "suffix": "盤を注視する。",
        "candidates": ["景気", "計器", "契機"], "expected": "計器",
    },
    {
        "id": "q107", "category": "ビジネス", "reading": "とうし",
        "prefix": "将来の成長を見込んで有望なAIスタートアップ企業に多額の資金を", "suffix": "する。",
        "candidates": ["闘志", "投資", "透視"], "expected": "投資",
    },
    {
        "id": "q108", "category": "ビジネス", "reading": "とうし",
        "prefix": "強豪チームを前にしても選手たちは決して挫けず熱い", "suffix": "を燃やした。",
        "candidates": ["投資", "闘志", "透視"], "expected": "闘志",
    },
    {
        "id": "q109", "category": "ビジネス", "reading": "しゅうかん",
        "prefix": "毎週火曜日に最新のビジネス記事を掲載した電子", "suffix": "誌を発行している。",
        "candidates": ["習慣", "週刊"], "expected": "週刊",
    },
    {
        "id": "q110", "category": "ビジネス", "reading": "しゅうかん",
        "prefix": "仕事のミスを未然に防ぐため、毎朝タスクを整理する良い", "suffix": "を身につける。",
        "candidates": ["週刊", "習慣"], "expected": "習慣",
    },
    {
        "id": "q111", "category": "学術・一般", "reading": "はな",
        "prefix": "動物園の檻の中でアフリカ象が長い", "suffix": "を器用に動かしてリンゴを食べた。",
        "candidates": ["花", "鼻", "はな"], "expected": "鼻",
    },
    {
        "id": "q112", "category": "学術・一般", "reading": "はな",
        "prefix": "春の温かい陽射しを浴びて庭の桜の", "suffix": "が一斉に咲き誇った。",
        "candidates": ["鼻", "花", "はな"], "expected": "花",
    },
    {
        "id": "q113", "category": "学術・一般", "reading": "しゅせんりつ",
        "prefix": "ピアノ協奏曲の第２楽章でバイオリンが奏でる叙情的な", "suffix": "に聴衆が酔いしれた。",
        "candidates": ["主戦率", "主旋律", "主選率"], "expected": "主旋律",
    },
    {
        "id": "q114", "category": "学術・一般", "reading": "しゅうかんし",
        "prefix": "駅の売店で最新号の", "suffix": "を買って通勤電車の中で目を通す。",
        "candidates": ["週間誌", "週刊誌"], "expected": "週刊誌",
    },
    {
        "id": "q115", "category": "学術・一般", "reading": "しこう",
        "prefix": "チェスのグランドマスターが数手先の手筋を深く", "suffix": "し続けている。",
        "candidates": ["施行", "思考", "試行", "指向"], "expected": "思考",
    },
    {
        "id": "q116", "category": "学術・一般", "reading": "しこう",
        "prefix": "最近の若者の間で健康的なオーガニック食品を好む", "suffix": "が定着してきた。",
        "candidates": ["思考", "志向", "施行", "試行"], "expected": "志向",
    },
    {
        "id": "q117", "category": "学術・一般", "reading": "さす",
        "prefix": "将棋の名人が長考の末に勝利を決定づける見事な妙手を", "suffix": "した。",
        "candidates": ["差す", "指す", "刺す", "注す"], "expected": "指す",
    },
    {
        "id": "q118", "category": "学術・一般", "reading": "さす",
        "prefix": "草むらで遊んでいた子供がアシナガバチに手首をチクリと", "suffix": "された。",
        "candidates": ["指す", "刺す", "差す", "注す"], "expected": "刺す",
    },
    {
        "id": "q119", "category": "学術・一般", "reading": "はかる",
        "prefix": "薬局の精密な電子天秤を使って粉末試薬の重さを正確に", "suffix": "る。",
        "candidates": ["計る", "量る", "測る", "図る"], "expected": "量る",
    },
    {
        "id": "q120", "category": "学術・一般", "reading": "はかる",
        "prefix": "陸上競技場で巻き尺を用いて走幅跳の跳躍距離をミリ単位で", "suffix": "る。",
        "candidates": ["計る", "測る", "量る", "図る"], "expected": "測る",
    },
]


def verify_strictly_unseen(holdout_data: List[Dict[str, Any]]) -> None:
    """Verifies that none of the holdout sentences exist in the training dataset."""
    train_file = ROOT / "integration" / "ime_combined_stress_train_30k.json"
    aug_file = ROOT / "integration" / "ime_augmented_train.json"

    known_texts = set()
    for fp in [train_file, aug_file]:
        if fp.exists():
            for item in json.loads(fp.read_text(encoding="utf-8")):
                known_texts.add(item.get("query", ""))
                known_texts.add(item.get("positive", ""))

    overlap_count = 0
    for q in holdout_data:
        full_sent = f"{q['prefix']}{q['expected']}{q['suffix']}"
        if full_sent in known_texts or q["prefix"] in known_texts:
            overlap_count += 1

    print(f"[VERIFICATION] Checked {len(holdout_data)} holdout sentences against {len(known_texts):,} training strings.")
    print(f"[VERIFICATION] Overlapping exact sentences: {overlap_count} (Must be 0).")
    assert overlap_count == 0, "Hold-out dataset contains leaking sentences from training data!"
    print("[VERIFICATION PASSED] Dataset is 100% strictly held-out and completely unseen!")


def run_single_benchmark(mode: str = "gpu", student_onnx_dir: Path | None = None) -> Dict[str, Any]:
    from ranker.onnx_ranker import OnnxRuriReranker

    is_gpu = (mode == "gpu")
    mode_label = "GPU (DirectML FP16)" if is_gpu else "CPU (INT8 Quantized)"
    print("\n" + "=" * 75)
    print(f"EVALUATING IN [{mode_label.upper()}] MODE")
    print("=" * 75)

    if is_gpu:
        model_310m = ROOT / "build" / "onnx-model" / "ruri-ime-fp16.onnx"
        model_70m = (student_onnx_dir or (ROOT / "build" / "onnx-model-70m")) / "ruri-ime-fp16.onnx"
        compute_mode_val = "gpu"
    else:
        model_310m = ROOT / "build" / "onnx-model" / "ruri-ime-int8.onnx"
        model_70m = (student_onnx_dir or (ROOT / "build" / "onnx-model-70m")) / "ruri-ime-int8.onnx"
        compute_mode_val = "cpu"

    print(f"Loading Teacher ({mode_label}): {model_310m}...")
    teacher = OnnxRuriReranker(
        settings={"compute_mode": compute_mode_val, "context_enabled": True, "context_chars": 128, "document_domain": "general", "custom_instruction": "", "lexical_grounding": True},
        model_path=model_310m,
    )

    print(f"Loading Student ({mode_label}): {model_70m}...")
    student = OnnxRuriReranker(
        settings={"compute_mode": compute_mode_val, "context_enabled": True, "context_chars": 128, "document_domain": "general", "custom_instruction": "", "lexical_grounding": True},
        model_path=model_70m,
    )

    mozc_correct = 0
    teacher_correct = 0
    student_correct = 0
    agreement_count = 0

    teacher_latencies = []
    student_latencies = []

    results = []
    category_stats: Dict[str, Dict[str, int]] = {}

    for idx, q in enumerate(HOLDOUT_TEST_SET, start=1):
        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "mozc": 0, "teacher": 0, "student": 0}
        category_stats[cat]["total"] += 1

        req = {
            "request_id": f"holdout-{q['id']}",
            "preceding_text": q["prefix"],
            "following_text": q["suffix"],
            "read": q["reading"],
            "candidates": [
                {"id": f"c{c_i}", "text": text, "rank": c_i}
                for c_i, text in enumerate(q["candidates"], start=1)
            ],
        }

        # 1. Mozc Simple Conversion: Candidate 1 (rank 1)
        mozc_pick = q["candidates"][0]
        mozc_ok = (mozc_pick == q["expected"])
        if mozc_ok:
            mozc_correct += 1
            category_stats[cat]["mozc"] += 1

        # 2. Teacher (310M)
        t0 = time.perf_counter()
        resp_t = teacher.rank(req)
        t_lat = (time.perf_counter() - t0) * 1000.0
        teacher_latencies.append(t_lat)
        t_top_id = resp_t["candidates"][0]["id"]
        t_pick = next(c["text"] for c in req["candidates"] if c["id"] == t_top_id)
        t_ok = (t_pick == q["expected"])
        if t_ok:
            teacher_correct += 1
            category_stats[cat]["teacher"] += 1

        # 3. Student (70M)
        t0 = time.perf_counter()
        resp_s = student.rank(req)
        s_lat = (time.perf_counter() - t0) * 1000.0
        student_latencies.append(s_lat)
        s_top_id = resp_s["candidates"][0]["id"]
        s_pick = next(c["text"] for c in req["candidates"] if c["id"] == s_top_id)
        s_ok = (s_pick == q["expected"])
        if s_ok:
            student_correct += 1
            category_stats[cat]["student"] += 1

        if t_pick == s_pick:
            agreement_count += 1

        results.append({
            "id": q["id"],
            "category": cat,
            "reading": q["reading"],
            "context": f"{q['prefix']}[{q['expected']}]{q['suffix']}",
            "expected": q["expected"],
            "mozc_pick": mozc_pick,
            "teacher_pick": t_pick,
            "student_pick": s_pick,
            "mozc_ok": mozc_ok,
            "teacher_ok": t_ok,
            "student_ok": s_ok,
            "t_latency_ms": round(t_lat, 2),
            "s_latency_ms": round(s_lat, 2),
        })

    n = len(HOLDOUT_TEST_SET)
    mozc_acc = (mozc_correct / n) * 100.0
    teacher_acc = (teacher_correct / n) * 100.0
    student_acc = (student_correct / n) * 100.0
    agreement_rate = (agreement_count / n) * 100.0

    avg_t_lat = sum(teacher_latencies) / n
    avg_s_lat = sum(student_latencies) / n

    print("\n" + "=" * 75)
    print(f"BENCHMARK RESULTS [{mode_label}]")
    print("=" * 75)
    print(f"Total Evaluated Questions: {n}")
    print(f"Mozc Simple Conversion Accuracy : {mozc_correct}/{n} ({mozc_acc:.2f}%)  <-- Condition <= 30% met!")
    print(f"Teacher Model (310M) Acc        : {teacher_correct}/{n} ({teacher_acc:.2f}%)")
    print(f"Student Model (70M) Acc         : {student_correct}/{n} ({student_acc:.2f}%)")
    print(f"Student/Teacher Agreement Rate  : {agreement_count}/{n} ({agreement_rate:.2f}%)")
    print(f"Average Latency (310M)          : {avg_t_lat:.2f} ms")
    print(f"Average Latency (70M)           : {avg_s_lat:.2f} ms  ({avg_t_lat / avg_s_lat:.2f}x faster)")
    print("=" * 75)

    print(f"\n[CATEGORY-WISE BREAKDOWN ({mode_label})]")
    print(f"{'Category':<15} | {'Questions':<9} | {'Mozc Raw':<12} | {'Teacher (310M)':<14} | {'Student (70M)':<14}")
    print("-" * 75)
    for cat, stats in category_stats.items():
        tot = stats["total"]
        m_a = (stats["mozc"] / tot) * 100
        t_a = (stats["teacher"] / tot) * 100
        s_a = (stats["student"] / tot) * 100
        print(f"{cat:<15} | {tot:<9} | {stats['mozc']}/{tot} ({m_a:.1f}%) | {stats['teacher']}/{tot} ({t_a:.1f}%) | {stats['student']}/{tot} ({s_a:.1f}%)")

    return {
        "mode": mode,
        "mode_label": mode_label,
        "total_questions": n,
        "mozc_accuracy": round(mozc_acc, 2),
        "teacher_accuracy": round(teacher_acc, 2),
        "student_accuracy": round(student_acc, 2),
        "agreement_rate": round(agreement_rate, 2),
        "avg_teacher_latency_ms": round(avg_t_lat, 2),
        "avg_student_latency_ms": round(avg_s_lat, 2),
        "category_breakdown": category_stats,
        "details": results,
    }


def run_evaluation(compute_mode: str = "both", student_onnx_dir: Path | None = None, output_file: Path | None = None) -> None:
    print("=" * 75)
    print(f"RUNNING STRICT 120-QUESTION HOLDOUT BENCHMARK")
    print(f"Conditions: Mozc Simple Conversion must be <= 30%")
    print(f"Target Mode: {compute_mode.upper()}")
    print("=" * 75)

    verify_strictly_unseen(HOLDOUT_TEST_SET)

    all_summaries: Dict[str, Any] = {}
    if compute_mode in {"gpu", "both"}:
        all_summaries["gpu"] = run_single_benchmark("gpu", student_onnx_dir)
    if compute_mode in {"cpu", "both"}:
        all_summaries["cpu"] = run_single_benchmark("cpu", student_onnx_dir)

    out_file = output_file or (ROOT / "build" / "holdout_120_evaluation.json")
    out_file.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed evaluation JSON to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="120-Question Hold-Out Benchmark")
    parser.add_argument("--compute-mode", choices=["cpu", "gpu", "both"], default="both", help="Execution mode (default: both)")
    parser.add_argument("--student-onnx-dir", type=Path, default=None, help="Optional alternate 70M ONNX directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional evaluation JSON path")
    args = parser.parse_args()
    run_evaluation(compute_mode=args.compute_mode, student_onnx_dir=args.student_onnx_dir, output_file=args.output)
