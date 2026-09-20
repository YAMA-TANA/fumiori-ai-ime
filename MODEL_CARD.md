# Fumiori AI IME — Model card

## Summary

Fumiori AI IMEはMozcが生成した日本語変換候補をローカルで再順位付けします。文章生成モデルではありません。**v2.1.1-betaの標準ランタイムはDual-Encoder 70M**です。旧Cross-Encoder LoRAアンサンブルは既存モデル向けの互換フォールバックとして残しています。

## Current runtime: Dual-Encoder 70M (v2.1.1-beta)

- 70M級の文脈encoderと候補encoderを使い、各埋め込みベクトルの内積で候補を比較します。
- 入力中に候補ベクトルを先読みし、Mozc候補の**最大10件**を再順位付けします。
- 固定候補の埋め込み辞書を最大50万語までバックグラウンドでSQLiteに保存し、モデル内容のハッシュに応じて再利用します。
- 入力中のプリフェッチ、確定後の文脈更新、複数文節の順位付けをDual-Encoder経路で連携させています。
- AIが停止・失敗した場合はMozcの候補順にフォールバックします。
- CPU向けINT8とGPU向けFP16のONNXモデルを用います。実際の速度・精度はハードウェア、文脈、候補に依存します。

最新配布物・変更点・インストール方式は[GitHub Release v2.1.1-beta](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)、[リリースノート](RELEASE_NOTES_v2.1.1-beta.md)、[README](README.md)を参照してください。

## Legacy models / provenance

### Cross-Encoder LoRA3 + LoRA6 ensemble（旧v2.0系）

- Historical bundle version: `v2.0.6-beta-70m-ensemble`（[model-manifest.json](model-manifest.json)の参照先）。**このmanifestは現在のDual-Encoder標準ランタイムそのものを記述するものではありません。**
- Base student: `cl-nagoya/ruri-v3-70m` (ModernBERT architecture, about 70.1M parameters)。310M teacherから蒸留した旧系列です。
- Historical fine-tuning data: 38,355 contextual pairs。ONNX opset 18、CPU Dynamic INT8 / GPU FP16。
- 旧モデルは候補と文脈を組み合わせてバッチ推論するCross-Encoder構成です。現在のDual-Encoderの精度や速度として旧モデルのベンチマーク値を引用しないでください。

### High-capacity model（旧310M系）

- Historical bundle version: `v0.1.0`。Base model: `cl-nagoya/ruri-v3-reranker-310m`（315M parameters）。
- ONNX opset 18、CPU Dynamic INT8 / GPU FP16。旧実験・比較のための構成です。

大容量モデルはGit履歴の外で配布されます。モデルの由来や各ファイルのSHA-256は該当リリースおよびmanifestで確認してください。

## Intended use and limitations

日本語IME候補の文脈適合度を比較する用途です。医学・法律等の文書分野設定やカスタム指示は補助情報であり、専門家の判断や文章内容の正確性を保証しません。ベータモデルのため誤変換・偏り・不自然な順位付けがあり得ます。文脈が短い、正解候補がMozcにない、固有名詞が未収録といった場合は改善しないことがあります。候補の相対順位は事実性や安全性の判定ではありません。

## Privacy, attribution and license

推論は利用者のPC内で行い、モデル自身に通信機能はありません。モデル・インストーラーのダウンロードではGitHubへの通信が発生します。詳細は[PRIVACY.md](PRIVACY.md)をご覧ください。Ruriの基盤モデルのクレジットは名古屋大学のCL Research GroupおよびRuri著作者に帰属します。独自部分・第三者コンポーネントのライセンスは[NOTICE](NOTICE)と[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)を参照してください。
