# Yamatana Ruri IME reranker model card

## Summary

Yamatana AI IME v2の変換候補再順位付け用モデルです。`cl-nagoya/ruri-v3-70m` を310MのIME向けteacherから蒸留し、ONNXへ変換しています。文章生成には使用せず、Mozcが提示した全候補を1回のバッチforwardで採点します。

## Version and provenance

### 1. LoRA3 + LoRA6 Ensemble (Recommended: 70M Series)
- Bundle version: `v2.0.6-beta-70m-ensemble`
- Runtime uses a calibrated 25/75 ensemble of the LoRA3 and preceding-only LoRA6 variants.
- Student model: `cl-nagoya/ruri-v3-70m` (ModernBERT architecture, 70.1M parameters, 22.3% size of teacher)
- Distillation: Margin-MSE + Soft KL + Hard Margin compound distillation from 310M teacher
- Fine-tuning data: 38,355 contextual pairs (including cultural agency homophone verbs and IT inference/implementation contexts)
- Export: ONNX opset 18
- CPU artifacts: Dynamic INT8 (`ruri-ime-lora3-int8.onnx`, `ruri-ime-lora6-int8.onnx`)
- GPU artifacts: DirectML FP16 (`ruri-ime-lora3-fp16.onnx`, `ruri-ime-lora6-fp16.onnx`)
- Artifact refresh: the 70M ONNX and tokenizer artifacts shipped in this release
  were refreshed on 2026-09-09; exact file hashes are pinned in
  `model-manifest.json`.
- Validation agreement with teacher: **99.85%** (Task val acc: 99.75%)

### 2. High-Capacity Model (310M Series)
- Bundle version: `v0.1.0`
- Base model: `cl-nagoya/ruri-v3-reranker-310m` (315M parameters)
- LoRA parameters: rank 16, alpha 32, dropout 0.05; ModernBERT `Wqkv`, `Wo`, `Wi`
- Export: ONNX opset 18
- CPU artifact: dynamic INT8 (`ruri-ime-int8.onnx`, 317.66 MB)
- GPU artifact: FP16 (`ruri-ime-fp16.onnx`, 631.04 MB)

The bundle and every required file are pinned by SHA-256 in `model-manifest.json`. The model is published separately from Git history.

## Intended use

Japanese IME候補の文脈適合度を比較する用途です。医学・法律等の設定やカスタム指示は補助情報であり、専門家の判断や文章内容の正確性を保証しません。

## Limitations

- Betaモデルであり、誤変換、偏り、不自然な順位付けがあり得ます。
- 前後文脈が短い、候補に正解がない、固有名詞が未収録の場合は改善しません。
- モデル出力は候補間の相対順位であり、事実性や安全性の判定ではありません。
- DirectMLの利用可否はGPU、ドライバー、同梱ONNX Runtimeに依存します。

## Privacy

推論はローカルで実行され、モデル自身に通信機能はありません。

## Attribution and license

Base model copyright and credit belong to the CL Research Group in Nagoya, Japan and the Ruri authors. Base and Yamatana model modifications are distributed under Apache License 2.0. See `NOTICE` and `THIRD_PARTY_LICENSES.md`.
