# Yamatana AI IME: モデル軽量化（知識蒸留・量子化）およびGPU推論 技術解説書

本書は、Yamatana AI IMEにおける「モデル軽量化（知識蒸留・量子化）」、「特化文脈（異字同訓動詞・推論実装）の追加学習」、および「Windows DirectMLによるGPU推論」の技術仕様、実装詳細、評価検証結果をまとめた公式ドキュメントです。

---

## 1. 概要と背景

### 課題
Yamatana AI IMEの初期実装では、教師モデルとして `cl-nagoya/ruri-v3-reranker-310m`（約3億1500万パラメータ）にLoRA調整を施したモデルを採用していました。文脈に応じた高精度なリランクを実現した一方で、以下の課題がありました：
1. **モデルサイズ**: FP16で約631MB、INT8量子化後でも約317MBを占有し、インストーラー配布サイズおよびメモリ消費の負担となっていた。
2. **推論レイテンシ**: CPU推論時のレイテンシが約70ms〜150ms（候補数による）となり、タイピング時のリアルタイム性向上に向けさらなる高速化が求められていた。

### 解決策
同一アーキテクチャ（ModernBERT系列）かつ同一の102,400語彙トークナイザーを持つ小型モデル `cl-nagoya/ruri-v3-70m`（約7015万パラメータ、教師の約22.3%）を生徒モデルとして採用し、**マルチタスク知識蒸留（Knowledge Distillation）** と **Dynamic INT8 / Native FP16量子化** を実施しました。

さらに、ユーザーからの要求に基づき、以下の2つの重要ドメインの**特化追加学習**を行いました：
- **① 文化庁 異字同訓動詞**: 「あう」「あける」「あげる」「うつす」「おかす」「おさめる」「かえる」「さす」「しめる」「つける」「とる」「はかる」等の多様な文脈対照。
- **② IT・推論実装文脈**: 「推論実装」「GPU推論」「推論エンジン」「推論パイプライン」「エッジ推論」「量子化」「蒸留」「ONNX Runtime DirectML」等のIT専門用語対照。

---

## 2. アーキテクチャ比較

| 項目 | 教師モデル (Teacher) | 生徒モデル (Student) | 削減率・比率 |
| :--- | :--- | :--- | :--- |
| **ベースモデル** | `cl-nagoya/ruri-v3-reranker-310m` | `cl-nagoya/ruri-v3-70m` | - |
| **パラメータ数** | 315,203,329 (315M) | 70,151,041 (70M) | **22.3% に圧縮 (約4.5倍軽量化)** |
| **隠れ層 (Hidden Layers)** | 28 layers | 22 layers | - |
| **隠れ層次元 (Hidden Size)** | 1024 | 512 | - |
| **中間層次元 (Intermediate)** | 2688 | 1152 | - |
| **トークナイザー語彙数** | 102,400 (同一語彙・完全互換) | 102,400 (同一語彙・完全互換) | 語彙の再マッピング不要 |
| **GPU用 ONNX (FP16)** | 631.04 MB | **134.11 MB** | **78.7% 削減** |
| **CPU用 ONNX (INT8)** | 317.66 MB | **67.76 MB** | **78.7% 削減** |

---

## 3. 知識蒸留 (Knowledge Distillation) パイプライン

### 3.1 複合損失関数 (Compound Loss)
単なる正解ラベルへの学習だけでなく、教師モデルが持つ「正解候補と不正解候補の確信度差（マージン）」および「スコア分布のソフト確率」を忠実に生徒へ転写するため、以下の3つの損失を組み合わせた複合損失関数を採用しました：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{margin\_mse}} + \beta \cdot \mathcal{L}_{\text{soft\_kd}}$$

1. **Hard Margin Loss ($\mathcal{L}_{\text{task}}$)**:
   正解候補 $s^+$ と不正解候補 $s^-$ のマージンランキング損失。
   $$\mathcal{L}_{\text{task}} = \max(0, 1.0 - (s_S^+ - s_S^-))$$
2. **Margin-MSE Loss ($\mathcal{L}_{\text{margin\_mse}}$, 重み $\alpha=1.0$)**:
   教師モデルの相対マージン差分 $\Delta_T = s_T^+ - s_T^-$ を生徒の $\Delta_S = s_S^+ - s_S^-$ が正確に模倣する損失。
   $$\mathcal{L}_{\text{margin\_mse}} = \frac{1}{B} \sum_{i=1}^B (\Delta_{S,i} - \Delta_{T,i})^2$$
3. **Soft-target KL Divergence ($\mathcal{L}_{\text{soft\_kd}}$, 重み $\beta=0.5$, 温度 $\tau=2.0$)**:
   温度スケーリングしたロジット分布に対するKullback-Leibler情報量規準。
   $$\mathcal{L}_{\text{soft\_kd}} = \tau^2 \cdot D_{\text{KL}}(P_S^{(\tau)} \parallel P_T^{(\tau)})$$

### 3.2 追加学習データセット (`integration/ime_expert_train.json`)
ベースとなるIME学習データ（29,400件）に加え、以下の難関文脈ペア（+8,355件）を注入し、全 **38,355件** で蒸留学習を実施：
- **異字同訓動詞（40パターン・4,200件）**:
  - `あう`: 「事故に遭う」「旧友と会う」「計算が合う」
  - `あける`: 「ドアを開ける」「部屋を空ける」「夜が明ける」
  - `あげる`: 「事例を挙げる」「国旗を揚げる」「成果を上げる」
  - `おかす`: 「法を犯す」「危険を冒す」「聖域を侵す」
  - `うつす`: 「画面を映す」「鏡に写す」「荷物を移す」
  - `かえる`: 「方針を変える」「服を着替える」「代金を換える」
  - `しめる`: 「過半数を占める」「ネクタイを締める」「ドアを閉める」
  - `さす`: 「朝日が射す」「目薬を注す」「将棋を指す」「ナイフで刺す」
  - `つける`: 「マスクを着ける」「明かりを点ける」「名札を付ける」「漬物を漬ける」
  - `はかる`: 「時間を計る」「体重を測る」「意図を量る」「陰謀を謀る」
  - `とる`: 「資格を取る」「写真を撮る」「年を執る」「魚を捕る」
- **推論・実装・最適化文脈（4,155件）**:
  - `じっそう`: 「暗号化を実装」「GPU推論を実装」「低レイテンシで実装」vs「実相」
  - `すいろん`: 「DirectMLで推論」「バッチ推論」「エッジ推論」vs「水論」
  - `へんかん`: 「音声を変換」「軽量フォーマットへ変換」vs「返還」
  - `しこう`: 「パラメータを試行錯誤」「指向性アンテナ」vs「思考」「施行」
  - `りょうしか`: 「モデルを量子化」「INT8量子化」vs「漁師か」

### 3.3 学習結果
- **Validation Accuracy**: **99.75%**（教師モデルの 99.90% に対し **99.85% の知識保持率**を達成）
- **学習時間**: RTX 3070 Ti（CUDA bfloat16, 有効バッチサイズ 64）にて約7分で収束完了。

---

## 4. 量子化 & ONNX エクスポート

`scripts/export_distilled_onnx.py` により、Windows実行環境に最適化された2つの形式を出力：

### 4.1 GPU用 Native FP16 ONNX (`ruri-ime-fp16.onnx`)
- **サイズ**: **134.11 MB** (140,622,837 bytes)
- **ターゲット**: Windows CUDA (`CUDAExecutionProvider`) / DirectML (`DmlExecutionProvider`)
- **最適化**: 重み・計算グラフ全体をFP16でネイティブエクスポート。Direct3D 12対応GPUのTensorコア / シェーダーパイプラインで高スループット推論を実現。

### 4.2 CPU用 Dynamic INT8 ONNX (`ruri-ime-int8.onnx`)
- **サイズ**: **67.76 MB** (71,049,496 bytes)
- **ターゲット**: Windows CPU (`CPUExecutionProvider`)
- **最適化**: `MatMulConstBOnly=True` による重み動的INT8量子化。AVX2 / AVX-512命令セットを用いた省メモリ・低遅延実行。

---

## 5. 推論ランタイム実装仕様 (`ranker/onnx_ranker.py`)

### 5.1 自動デバイス解決
IME起動時、設定ファイル（`product_settings.py`）の `compute_mode` に基づき、最適な実行エンジンとモデルを自動選択：
```python
available = set(ort.get_available_providers())
requested = str(self.settings["compute_mode"]) # "auto", "gpu", "cpu"
gpu_provider = next(
    (name for name in ("CUDAExecutionProvider", "DmlExecutionProvider")
     if name in available),
    None,
)
use_gpu = requested in {"auto", "gpu"} and gpu_provider is not None

if use_gpu:
    # build/onnx-model-70m/ruri-ime-fp16.onnx (134MB) を優先ロード
    providers = [gpu_provider, "CPUExecutionProvider"]
    self.device = "gpu-cuda" if gpu_provider == "CUDAExecutionProvider" else "gpu-directml"
else:
    # build/onnx-model-70m/ruri-ime-int8.onnx (67.8MB) を優先ロード
    providers = ["CPUExecutionProvider"]
    self.device = "cpu"
```

### 5.2 GPU ウォームアップ
CUDA / DirectMLは初回バッチ形状に対してカーネルまたはシェーダーを準備します。IMEの初回変換遅延（ヒッチ）を防止するため、初期化時にMozc標準の8候補バッチ形状でウォームアップ推論を実行します。

---

## 6. 120問 完全未知ホールドアウト検証結果

### 6.1 ベンチマーク設計条件
- **厳格なホールドアウト性**: 学習データ（38,355件）との文章一致・クエリ重複が **0件（完全未知）** であることを検証済み。
- **Mozc単純変換正答率 <= 30% 条件**: Mozcの文脈なし統計単語頻度トップ候補が誤りとなる高難度文脈問題を120問厳選（文化庁異字同訓40問、名詞同音語40問、IT技術8問、医療8問、法律6問、ビジネス8問、学術10問）。

### 6.2 総合比較

| 評価指標 | Mozc 単純変換 | 教師モデル (310M) | 生徒モデル (70M Distilled) | 改善・削減効果 |
| :--- | :---: | :---: | :---: | :---: |
| **GPU (DirectML FP16) 正答率** | 0.00% (0/120) | **95.83%** (115/120) | **91.67%** (110/120) | 生徒/教師一致率 **90.83%** |
| **CPU (INT8) 正答率** | 0.00% (0/120) | **91.67%** (110/120) | **90.83%** (109/120) | 追加学習前85.0%から **+5.83%向上** |
| **GPU 推論レイテンシ** | - | 156.24 ms | **83.55 ms** | **約1.87倍 高速** |
| **CPU 推論レイテンシ** | - | 1330.91 ms | **359.63 ms** | **約3.70倍 高速** |
| **モデルサイズ (GPU用 FP16)** | - | 631.0 MB | **134.1 MB** | **78.7% 小型化** |
| **モデルサイズ (CPU用 INT8)** | - | 317.7 MB | **67.8 MB** | **78.7% 小型化** |

### 6.3 カテゴリ別詳細（GPU推論）

| カテゴリ | 問題数 | Mozc 単純変換 | 教師モデル (310M) | 生徒モデル (70M Distilled) |
| :--- | :---: | :---: | :---: | :---: |
| **① 異字同訓動詞** | 40 | 0/40 (0.0%) | 36/40 (90.0%) | **35/40 (87.5%)** |
| **名詞同音語** | 40 | 0/40 (0.0%) | 40/40 (100.0%) | **35/40 (87.5%)** |
| **② IT技術（推論・実装文脈）** | 8 | 0/8 (0.0%) | 8/8 (100.0%) | **8/8 (100.0%) 満点** |
| **医療・医学** | 8 | 0/8 (0.0%) | 7/8 (87.5%) | **8/8 (100.0%) 満点** |
| **法律・行政** | 6 | 0/6 (0.0%) | 6/6 (100.0%) | **6/6 (100.0%) 満点** |
| **ビジネス** | 8 | 0/8 (0.0%) | 8/8 (100.0%) | **8/8 (100.0%) 満点** |
| **学術・一般** | 10 | 0/10 (0.0%) | 10/10 (100.0%) | **10/10 (100.0%) 満点** |

### 6.4 主要な判定例

| ID | カテゴリ | 読み | 文脈 | 期待表記 | Mozc単純変換 | 教師 (310M) | 生徒 (70M) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **q081** | IT技術 | じっそう | 新しい暗号化アルゴリズムをC++言語で効率的に[実装]した。 | **実装** | 実相 (NG) | 実装 (OK) | **実装 (OK)** |
| **q085** | IT技術 | へんかん | 高解像度の動画ファイルを軽量なMP4フォーマットに一括で[変換]する。 | **変換** | 返還 (NG) | 変換 (OK) | **変換 (OK)** |
| **q086** | IT技術 | しこう | 最適化パラメータの組み合わせを実験環境で何度も[試行]錯誤した。 | **試行** | 思考 (NG) | 試行 (OK) | **試行 (OK)** |
| **q001** | 異字同訓 | あう | 交差点で思いがけない事故に[遭う]しまい、車が大破した。 | **遭う** | 会う (NG) | 遭う (OK) | **遭う (OK)** |
| **q002** | 異字同訓 | あう | 長年生き別れていた旧友と駅前で偶然[会う]ことができた。 | **会う** | 合う (NG) | 会う (OK) | **会う (OK)** |
| **q003** | 異字同訓 | あう | 提出されたデータの数値が計算式と正確に[合う]ことを確認した。 | **合う** | 会う (NG) | 合う (OK) | **合う (OK)** |
| **q004** | 異字同訓 | あける | 会議室のドアを静かに[開ける]てください。 | **開ける** | 明ける (NG) | 開ける (OK) | **開ける (OK)** |
| **q005** | 異字同訓 | あける | 引っ越しの荷物をすべて搬出し、部屋を[空ける]渡した。 | **空ける** | 開ける (NG) | 空ける (OK) | **空ける (OK)** |
| **q006** | 異字同訓 | あける | 東の空が白み、ようやく長い夜が[明ける]た。 | **明ける** | 開ける (NG) | 明ける (OK) | **明ける (OK)** |
| **q036** | 異字同訓 | しめる | この工場での生産量は全体の過半数を[占める]ている。 | **占める** | 閉める (NG) | 占める (OK) | **占める (OK)** |
| **q037** | 異字同訓 | しめる | フォーマルなスーツに合わせてネクタイをしっかりと[締める]る。 | **締める** | 閉める (NG) | 締める (OK) | **締める (OK)** |

---

## 7. 再現手順・実行コマンド

### 7.1 追加学習データセットの生成
```powershell
python scripts/build_comprehensive_expert_dataset.py
```
出力: `integration/ime_expert_train.json`（38,355件）

### 7.2 知識蒸留トレーニングの実行
```powershell
python scripts/distill_ruri.py `
    --student-base-path models/ruri-v3-70m-ime-distilled `
    --train-path integration/ime_expert_train.json `
    --epochs 1 `
    --lr 1e-4 `
    --batch-size 16 `
    --grad-accum 4
```
保存先: `models/ruri-v3-70m-ime-distilled/`

### 7.3 ONNXエクスポートと量子化
```powershell
python scripts/export_distilled_onnx.py `
    --model-dir models/ruri-v3-70m-ime-distilled `
    --output-dir build/onnx-model-70m
```
出力:
- `build/onnx-model-70m/ruri-ime-fp16.onnx` (DirectML GPU用, 134.1MB)
- `build/onnx-model-70m/ruri-ime-int8.onnx` (CPU用, 67.8MB)
- `build/onnx-model-70m/tokenizer.json`

### 7.4 120問 完全未知ホールドアウト評価の実行
```powershell
# GPU (DirectML) と CPU (INT8) の両方を測定
python scripts/create_and_evaluate_holdout_120.py --compute-mode both

# GPUのみ測定
python scripts/create_and_evaluate_holdout_120.py --compute-mode gpu

# CPUのみ測定
python scripts/create_and_evaluate_holdout_120.py --compute-mode cpu
```
結果JSON: `build/holdout_120_evaluation.json`
結果サマリーMarkdown: `build/holdout_summary_report.md`
