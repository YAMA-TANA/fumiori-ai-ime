# Fumiori AI IME

**文脈に合わせて、Mozcの変換候補を並べ替えるWindows向け日本語IME。** Fumiori AI IMEは、Mozcが作った候補をローカルAIで再順位付けします。文章を自由生成する仕組みではなく、入力内容・前後文脈・AI推論をPC内で処理します。

> **公開ベータ / 未署名** — 現在の配布物は検証用の未署名ビルドです。Windowsの発行元・SmartScreen警告が出る場合があります。署名済みであるとは表示しません。コード署名については[方針](CODE_SIGNING_POLICY.md)をご確認ください。

> **最新の公開ベータ: [v2.1.1-beta](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)** — 新しいPCに直接インストールできる**モデル同梱のWindows x64 MSI**を公開しました。旧名称を含む配布ファイル名は互換性のため維持しています。

[**公式サイト・変換デモ**](https://yama-tana.github.io/fumiori-ai-ime/) · [**フルMSIをダウンロード**](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Yamatana-AI-IME-MOZC-Ver-2.1.1-beta-x64.msi) · [**リリースノート**](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta) · [**不具合報告**](https://github.com/YAMA-TANA/fumiori-ai-ime/issues)

## どんなIME？

```text
彼の顔のはな             → 鼻
庭には美しいはな         → 花
役員が稟議書をけっさい   → 決裁
オンラインで代金をけっさい → 決済
明日の天気予報はあめだ   → 雨
弟にお土産としてあめを買う → 飴
```

同じ読みでも、文脈に適した候補を先頭へ持ってくることを目指しています。候補自体はMozcが生成し、AIが失敗・停止した場合はMozcの候補順に戻ります。変換結果は文脈や候補の有無によって異なります。

![Fumiori AI IMEの変換デモ](pr-video/ai-ime-demo.gif)

## v2.1.1-betaの主な特徴

- **Dual-Encoder 70Mが標準**：文脈と候補を別々にベクトル化し、内積で候補を比較。入力中に候補ベクトルを先読みし、明示変換ではMozc候補の最大10件を再順位付けします。
- **候補ベクトルを再利用**：固定候補の埋め込みを最大50万語までバックグラウンドでSQLiteに保持し、次回以降に活用します。
- **複数文節・文脈更新に対応**：Dual-Encoder経路で変換と確定後の文脈更新を連携させています。
- **ローカル処理**：入力、前後文脈、カスタム指示、辞書、AI推論をPC内で処理。外部AI APIへの送信やテレメトリは実装していません。
- **普段の入力を維持**：トレイからAI ON/OFF、文脈保持、文書分野、カスタム指示、CPU/GPU設定を変更できます。AI処理に失敗した際はMozcへフォールバックします。
- 旧Cross-Encoder LoRAアンサンブルは、既存モデルしかない環境向けの互換フォールバックとして残しています。

Dual-Encoderは候補ごとに文脈を含めてモデルを再実行するCross-Encoderと異なり、文脈ベクトルを共有し、候補側の計算結果も再利用できます。実際の速度・精度はPC環境や変換内容に依存します。

## 動作環境

- Windows 10 22H2（build 19045）またはWindows 11、x64
- RAM 8GB以上（16GB推奨）、空き容量約3GB
- CPU推論に対応。対応GPUではCUDA、次にDirectMLを優先する構成です。
- インストールには管理者権限が必要です。

[システム要件](docs/SYSTEM_REQUIREMENTS_JA.md) · [モデル・GPU技術解説](docs/DISTILLATION_QUANTIZATION_AND_GPU_INFERENCE_JA.md)

## インストール

### 新しいPCにインストールする場合（推奨）

1. [**v2.1.1-beta フルMSI（Windows x64）**](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Yamatana-AI-IME-MOZC-Ver-2.1.1-beta-x64.msi)をGitHub Releaseからダウンロードします。**モデルを同梱した完全版**で、旧v2.1.0 MSIや更新ZIPを別途ダウンロードする必要はありません。
2. [リリースの `SHA256SUMS.txt`](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/SHA256SUMS.txt)とファイルのハッシュを照合し、信頼できる場合にMSIを実行して管理者権限を許可します。PowerShellでは `Get-FileHash .\Yamatana-AI-IME-MOZC-Ver-2.1.1-beta-x64.msi -Algorithm SHA256` で確認できます。
3. インストール完了後、サインアウトまたは再起動します。サインイン後に `Win + Space` で **Fumiori AI IME** を選択してください。AIは初期ONで、通知領域からOFFにできます。

MSIの公開SHA-256：`43B919E448F1A57351BB7D6FA4361C32589A8B375775DE4E23DCBD84C1997D16`

未署名のベータ版です。Windowsの警告を無条件に回避せず、配布元・ハッシュを確認して判断してください。リリースには補助用の `install-msi.ps1` と `Install-Yamatana-AI-IME.cmd` もありますが、**新規インストールは上記MSI単体で可能**です。

### 以前のインストールがある場合

[同じリリース](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` は**既存インストール向け差分更新**です。展開して `scripts\apply_hotfix.ps1` を管理者PowerShellで実行する方式で、ZIP単体では新規インストールできません。既存環境にフルMSIを適用する場合は、環境に応じた更新・アンインストール手順を確認してください。

以前の[新規PC用セットアップZIP](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Fumiori-AI-IME-New-PC-Setup-v2.1.1-beta.zip)も公開されていますが、こちらは旧v2.1.0のMSIと差分更新を順番に取得する方式です。**現在の推奨ダウンロードは上記フルMSI**です。本製品はMSI配布です。

### アンインストール

Windowsの **設定 → アプリ → インストールされているアプリ → Fumiori AI IME → アンインストール** を選び、必要に応じてサインアウトまたは再起動してください。

## 評価結果について

過去バージョンの実際のMozc候補を用いた210例の検証では、実用正解率がMozc単体72.4%、当時のAI併用99.1%でした。**これは過去の特定データ・モデル構成での検証値であり、現行Dual-Encoderの一般的な変換精度や保証値ではありません。** モデルと検証条件が異なる数字を直接比較しないでください。現行方式の詳細は[リリースノート](RELEASE_NOTES_v2.1.1-beta.md)と[モデルカード](MODEL_CARD.md)を参照してください。

## プライバシー・開発・ライセンス

入力内容、文脈、辞書、推論結果を外部へ送信せず、テレメトリも実装していません。**GitHubからインストーラー・モデルをダウンロードする際の通信**はGitHub側の通常のアクセスログ・プライバシーポリシーの対象です。詳しくは[PRIVACY.md](PRIVACY.md)をご覧ください。

大容量ONNXモデルはGit履歴に含めず、[model-manifest.json](model-manifest.json)と[scripts/fetch-model.ps1](scripts/fetch-model.ps1)で取得・ハッシュ検証します。Mozcの固定fork/commitは[build-config.json](build-config.json)を参照してください。

開発参加：[CONTRIBUTING.md](CONTRIBUTING.md) · セキュリティ報告：[SECURITY.md](SECURITY.md) · 独自部分のライセンス：[Apache-2.0](LICENSE) · 上流ライセンス：[NOTICE](NOTICE) / [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) · コード署名：[CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md)
