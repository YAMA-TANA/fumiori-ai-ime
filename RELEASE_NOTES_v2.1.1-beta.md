# Fumiori AI IME v2.1.1-beta

## Dual-Encoderを標準ランタイムに変更

- Dual-Encoder 70Mを標準の候補再順位付けエンジンにしました。
- 文脈ベクトルと候補ベクトルを分離し、候補ベクトルを先読みして内積で比較します。
- 旧Cross-Encoder LoRAアンサンブルは、既存モデルしかない環境の互換フォールバックとして残しています。

## 変換経路の修正

- 入力中のプリフェッチ、確定後の文脈更新、複数文節の一括順位付けをDual-Encoder経路でつなぎました。
- Mozcが生成する候補のうち最大10件をAIへ渡し、AI停止・失敗時はMozcの候補順に戻ります。
- 10件対応の送信側とAIランタイムを同じビルドへ揃え、候補数上限の不一致によるリクエスト破棄を防ぎました。
- 固定候補の埋め込み辞書を最大50万語までバックグラウンドで事前計算し、モデル内容のハッシュで再利用します。

## 新規PC向けフルMSI（追加配布）

現在は[GitHub Release](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)に、モデルを同梱した**Windows x64フルMSI**を公開しています。旧v2.1.0のMSIや差分ZIPを別途取得しなくても新規インストールできます。

- [Yamatana-AI-IME-MOZC-Ver-2.1.1-beta-x64.msi](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Yamatana-AI-IME-MOZC-Ver-2.1.1-beta-x64.msi) — 新規インストール用の完全版。ファイル名のYamatanaは旧内部名です。
- [SHA256SUMS.txt](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/SHA256SUMS.txt) — ダウンロードしたMSIの整合性確認用。
- `install-msi.ps1` / `Install-Yamatana-AI-IME.cmd` — リリース内の補助スクリプト。

MSI SHA-256: `43B919E448F1A57351BB7D6FA4361C32589A8B375775DE4E23DCBD84C1997D16`

**公開ベータは未署名です。** Windowsの警告が出る場合があります。配布元とハッシュを確認してからインストールしてください。インストール後はサインアウトまたは再起動し、`Win + Space`でFumiori AI IMEを選択してください。

## 既存インストール向け差分更新

- `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` — 既存インストール向けの差分更新です。`scripts\apply_hotfix.ps1`を管理者PowerShellで実行します。**ZIP単体では新規インストールできません。**
- [以前の新規PC用セットアップZIP](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Fumiori-AI-IME-New-PC-Setup-v2.1.1-beta.zip) — `Install-Fumiori-AI-IME.cmd`と`install-fresh.ps1`を含み、旧v2.1.0モデル同梱MSIと差分更新をSHA-256照合のうえ順番に適用する従来方式です。新規PCでは上記フルMSIを優先してください。

変更の反映にはサインアウトまたは再起動が必要です。最新のインストール手順は[README](README.md#インストール)をご覧ください。
