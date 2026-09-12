# 同音異義語データ収集（再現可能な構成）

LoRA残差学習用の同音異義語データは、手作りテンプレートだけに依存しないよう、次の公開資源から再生成できる形にした。

| 資源 | 収集内容 | ライセンス／注意 |
|---|---|---|
| [JMdict / EDRDG](https://www.edrdg.org/jmdict/j_jmdict.html) | 読みごとの見出し表記、語義グロス。今回の取得時点で同一読み2語以上のグループを44,358組抽出 | EDRDG JMdict licence。配布物には出典・ライセンス表示を残す |
| [Tatoeba 日本語文](https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2) | 実際の日本語文から、同一読みグループ内の表記を置換した自然文hard-negative | CC BY 2.0 FR。文データの再配布条件と帰属表示を守る |
| [文化庁「異字同訓」の漢字の用法](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/sanko/yohorei/index.html) | 公的な使い分けの基準。curatedテンプレートの根拠 | 政府公開資料。出典表示を行う |

## 生成方法

```powershell
python scripts/collect_comprehensive_homophone_data.py --max-pairs 30000 --max-per-word 30
python scripts/build_residual_lora_dataset.py
```

前者は `data/homophone_sources/manifest.json` に取得URLとSHA-256、抽出グループ数、生成ペア数を記録する。ダウンロードした原データと生成JSONLは `.gitignore` 対象で、モデル配布物へ含めない。後者は、これらの自然文ペアを既存の異字同訓テンプレート、ひらがな保護例、一般日本語replayと混ぜる。

今回の取得結果（2026-09-09 JST）は、JMdict 44,358グループ、Tatoeba置換ペア30,000件、LoRA入力143,545ペア。holdout文そのものは生成テンプレートへコピーしていない。
