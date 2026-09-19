# 同音異義語LoRA改善 実験記録（2026-09-09）

## データ

- curatedな実務・異字同訓テンプレートに、[JMdict/EDRDG](https://www.edrdg.org/jmdict/j_jmdict.html) の読みグループと [Tatoeba日本語文](https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2) の自然文置換ペアを追加。
- JMdictの同一読み2語以上のグループ: 44,358組。
- Tatoeba自然文 hard-negative: 30,000件。
- 第1ラウンド入力: 143,545ペア。holdout本文との完全一致は全生成JSON（5ファイル）を横断して0件。
- 残差ラウンドでは、holdoutで落ちた読みを別文脈で追加（ラウンド2 50,315、ラウンド3 40,800、ラウンド4 21,000、ラウンド5 37,600）。

## strict holdout 120問（CPU INT8）

| 版 | 正解 | 精度 | 備考 |
|---|---:|---:|---|
| 更新済み70M（LoRA前） | 105/120 | 87.50% | v2.0.5-betaで評価した基準 |
| LoRAラウンド1 | 109/120 | 90.83% | JMdict/Tatoeba + curated |
| LoRAラウンド2 | 112/120 | 93.33% | 残差読み追加 |
| LoRAラウンド3 | 115/120 | 95.83% | 8語の高精度残差 |
| LoRAラウンド4 | **116/120** | **96.67%** | 単体モデルの最良 |
| LoRAラウンド5 | 115/120 | 95.83% | 狭い形態残差で一部回帰 |
| ラウンド3 + 4 の evidence-score平均 | **120/120** | **100.00%** | 2モデルensemble、同じ未見holdout |

ラウンド4単体の平均INT8レイテンシは18.48ms。ensembleは2本の推論を行うため、おおむね2倍の推論コストになる。したがって現時点では、単体配布候補はラウンド4、98%超を狙う実装候補は2本ensembleとして扱い、別の新規holdoutで再確認してから製品版へ昇格させる。

## 新規holdout 100問（2026-09-09、CPU INT8）

既存120問、全`integration/*.json`訓練シャード、元のholdout本文・prefixと完全一致しない100文を新規作成し、評価前にリーク検査を実施した（完全一致0件）。

| 版 | 正解 | 精度 |
|---|---:|---:|
| LoRAラウンド4単体 | 95/100 | 95.00% |
| ラウンド3 + 4 の evidence-score平均 | 96/100 | 96.00% |

目標98%には未達だった。ensembleで改善したのは1問で、残りの誤りは「あう→合う」「空ける」「刺す」「付ける」「端」など、異字同訓・表記選択に集中している。このholdoutは学習へ戻しておらず、測定専用として保持する。

再現コマンド:

```powershell
python scripts/evaluate_new_holdout.py
```

## 実務文・助詞境界ベンチ（2026-09-09、CPU INT8）

助詞を含めない変換（`はな` + 後続`が`）と、助詞込みの変換（`はなが` → `鼻が`/`花が`）を同じ56文で比較した。さらに、Mozc 1位が正しい12文をcontrolとして追加し、AIが不要に別候補へ切り替えるfalse switchも数えた。全136件は訓練シャードと旧120問に対して完全一致0件だった。

| 群・分割 | LoRA4単体 | LoRA3+4 ensemble |
|---|---:|---:|
| stress・助詞別セグメント | 49/56 (87.50%) | 49/56 (87.50%) |
| stress・助詞込み | 49/56 (87.50%) | 51/56 (91.07%) |
| control・助詞別セグメント | 11/12 (91.67%) | 11/12 (91.67%) |
| control・助詞込み | 11/12 (91.67%) | 11/12 (91.67%) |

controlの不要切替は単体2/24、ensemble 2/24。代表的な誤りは「窓辺の花」を「鼻」にする切替で、stress側は「開ける/空ける」「替える/変える」「刺す/指す」「改訂/改定」「解答/回答」に集中した。助詞込みで改善はあるが、98%にはまだ届かず、助詞境界を安全補正の独立特徴として扱う必要がある。

再現コマンド:

```powershell
python scripts/evaluate_practical_particle_holdout.py
```

## 前文脈のみ・変換範囲ベンチ（2026-09-09、CPU INT8）

実運用に合わせ、全152件で`following_text`を空にした。56文を単語範囲と単語＋助詞範囲で各1回、さらに長い一セグメント（例: `製品の仕様を`）12件と、Mozc 1位が正しいcontrol 28件を評価した。

| 群・範囲 | LoRA4単体 | LoRA3+4 ensemble |
|---|---:|---:|
| stress・単語 | 42/56 (75.00%) | 46/56 (82.14%) |
| stress・単語＋助詞 | 42/56 (75.00%) | 46/56 (82.14%) |
| stress・長い一セグメント | 8/12 (66.67%) | 8/12 (66.67%) |
| control・単語 | 10/12 (83.33%) | 10/12 (83.33%) |
| control・単語＋助詞 | 10/12 (83.33%) | 10/12 (83.33%) |
| control・長い一セグメント | 3/4 (75.00%) | 1/4 (25.00%) |

前文脈だけでは、前回の右文脈付き結果より大幅に低下した。特に長い一セグメントは、単語表記の候補が長い句に埋め込まれていても8/12に留まる。これは「4文字だからAIを止める」だけの問題ではなく、AIに渡す文脈が前方だけになったときのモデル性能不足も示している。最終的な安全補正では、単語・助詞・長句の変換範囲を分け、controlで不要切替を抑える必要がある。

再現コマンド:

```powershell
python scripts/evaluate_preceding_only_range_holdout.py
```

## 前文脈専用LoRA6（2026-09-09）

前文脈のみの低下を補うため、別文脈の実務テンプレートを追加学習した。助詞別セグメント24,168ペア、単語セグメント24,024ペア、長い一セグメント950ペア、新規合計49,142ペアに、既存データのreplay約12,000件を加えた計61,142ペアである。学習はLoRA4を初期値、2 epoch、learning rate `7e-5` とし、holdout本文は完全一致除外した。

| 前文脈のみ stress | LoRA4 | LoRA6単体 | LoRA3+6 ensemble |
|---|---:|---:|---:|
| 単語 | 42/56 (75.00%) | 47/56 (83.93%) | 51/56 (91.07%) |
| 単語＋助詞 | 42/56 (75.00%) | 50/56 (89.29%) | 52/56 (92.86%) |
| 長い一セグメント | 8/12 (66.67%) | 9/12 (75.00%) | 9/12 (75.00%) |

旧strict 120問ではLoRA6単体114/120 (95.00%)、LoRA3+6 ensemble 120/120 (100.00%)。新規100問ではLoRA6単体92/100 (92.00%)、LoRA3+6 ensemble 97/100 (97.00%)だった。したがってLoRA6単体はround4単体を置き換えず、現時点では前文脈を重視する2モデルensemble候補として扱う。98%には新規100問であと1問届いていない。

生成・学習・評価コマンド:

```powershell
python scripts/build_preceding_only_lora_round6_dataset.py
python scripts/train_70m_residual_lora.py `
  --base-model build/ruri-v3-70m-ime-lora4-20260909 `
  --dataset integration/ime_residual_lora_round6_preceding_only.json `
  --output build/ruri-v3-70m-ime-lora6-preceding-only-20260915 `
  --epochs 2 --batch-size 32 --grad-accum 2 --lr 7e-5
python scripts/export_distilled_onnx.py `
  --model-dir build/ruri-v3-70m-ime-lora6-preceding-only-20260915 `
  --output-dir build/onnx-model-70m-lora6-preceding-only-20260915
```

再現コマンド:

```powershell
python scripts/collect_comprehensive_homophone_data.py --max-pairs 30000 --max-per-word 30
python scripts/build_residual_lora_dataset.py
python scripts/train_70m_residual_lora.py ...
python scripts/evaluate_lora_ensemble.py `
  --model-a build/onnx-model-70m-lora3-20260909/ruri-ime-int8.onnx `
  --model-b build/onnx-model-70m-lora4-20260909/ruri-ime-int8.onnx
```

この実験で作ったモデルは `build/` の一時成果物であり、既存のGitHub releaseやインストール済みモデルは置換していない。

## LoRA3+LoRA6 実ランナーのアンサンブル調整（2026-09-09）

評価器だけでスコアを平均するのではなく、ONNXランナー自身が2本のモデルを読み、raw logitsを統合してから表記補正と候補選択を一度だけ行う経路を追加した。

- 通常の読み（4文字以上）: LoRA3 25% + LoRA6 75%
- 短い読み（3文字以下）: LoRA3 50% + LoRA6 50%（モデルの意見が割れやすいため）
- 2モデル時は単体用のMozc順位ペナルティを0にし、単体用安全ゲートを自動適用しない。必要なら`safety_gate=True`で比較できる。
- 各候補の`model_scores`、使用重み、モデル一覧を説明ペイロードに保存する。

実ランナー経路（strict 120、新規100、前文脈のみ152）:

| ベンチ | 正解 | 精度 |
|---|---:|---:|
| strict 120 | 120/120 | 100.00% |
| 新規100 | 97/100 | 97.00% |
| 前文脈のみ152 | 135/152 | 88.82% |

起動例:

```powershell
python -m ranker.ranker --backend onnx --pipe ai_ime_ranker `
  --ensemble-model build/onnx-model-70m-lora3-20260909/ruri-ime-int8.onnx `
  --ensemble-model build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-int8.onnx
```

再現評価:

```powershell
python scripts/evaluate_runtime_ensemble.py `
  --model-a build/onnx-model-70m-lora3-20260909/ruri-ime-int8.onnx `
  --model-b build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-int8.onnx
```
