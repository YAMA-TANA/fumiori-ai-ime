# Yamatana AI IME v2.0.7-beta

## 起動修正

- MSIが登録するWindowsログオン時のトレイ起動経路に合わせ、初回設定のAI自動起動を既定ONへ変更。
- `--from-installer` が渡された場合も、初回起動をOFF状態にせずAIモデルを開始するよう修正。
- AIをOFFにした利用者の設定は保持し、トレイから従来どおり切り替え可能。

## 確認

- CUDA対応GPUでは、既定の自動選択でFP16モデルと `CUDAExecutionProvider` を使用。CUDAが利用できない環境ではDirectML、CPUの順にフォールバック。
- Mozc → named pipe → ONNXランナーの実変換経路で、候補の再順位付けとGPU実行を確認。

MSIの内部ProductVersionは `2.0.7.0`、公開リリース名は `v2.0.7-beta` です。未署名Betaのため、配布ハッシュを照合して使用してください。
