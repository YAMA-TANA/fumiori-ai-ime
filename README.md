# Fumiori AI IME

Fumiori AI IMEは、Mozcの変換候補をローカルAI rerankerで並べ替えるWindows向け日本語IMEです。文脈、文書分野、ユーザー辞書相当の語彙情報を変換判断に使いながら、入力内容を外部へ送信しません。

> **Beta / 未署名** — 現在公開中のBetaは検証用の未署名ビルドです。Windowsの警告が表示される場合があります。SignPath FoundationによるOSSコード署名の申請準備中であり、署名済みであるかのような表示は行いません。

## まず試してほしいこと

同じ読みでも、文脈で漢字が変わる日本語入力を、AIに選ばせるIMEです。

```text
彼の顔のはな       → 鼻
庭には美しいはな   → 花
役員が稟議書をけっさい → 決裁
オンラインで代金をけっさい → 決済
明日の天気予報はあめだ → 雨
弟に御土産としてあめを買う → 飴
```

Mozcが作った候補をAIが読み直すため、AIが勝手に文章を生成するのではなく、普段のMozcの安心感を残したまま「いちばん自然な候補」を先に出せます。

同じ「あめ」でも、天気の話なら「雨」、お土産の話なら「飴」。このような文脈による使い分けが、デモの見どころです。

実際のMozc候補を使った210例の検証では、自然な表記を含めた正解率が **72.4%（Mozcのみ）から99.1%（AI併用）** になりました。これは検証用データでの結果であり、製品全体の精度を保証する数字ではありません。条件・データ・スクリプトは公開しています。

### ドット絵ゲーム風の変換体験

変換候補ウィンドウは、8bitゲームを思わせる限定色、カクカクしたステップ角、選択中の「ヒット」カードで構成しています。AIが候補を入れ替えたときは `CHAIN +1` が表示され、文脈に合う候補を選ぶほど連鎖が続くように見えるデザインです。見た目はゲーム風でも、変換の確定操作とMozcのフェイルセーフは従来どおりです。

詳しい背景と実例はこちらです。

- [Qiita：AIに変換候補を選ばせるIMEを作ってみた](https://qiita.com/yamatana364/items/c9b640e64070a798c4ff)
- [Zenn：AIに変換候補を選ばせるIMEを作ってみた](https://zenn.dev/yamatana/articles/92cd6e19eb8d45)

### 変換候補が変わる様子

冒頭の「明日の天気予報は雨だ」の変換から、後半の「御土産」の候補選びまでを収録しています。Mozcの候補をAIが文脈に合わせて選び、確定候補を入れ替える流れです。

![AI-IMEの変換候補デモ](pr-video/ai-ime-demo.gif)

## 特徴

- Mozcベースの通常変換を保ったまま、AI有効時だけ候補を再順位付け
- 70M級Ruri v3 student rerankerをIME向けに蒸留し、全候補を1バッチでONNX Runtime実行
- 入力、前後文脈、カスタム指示、辞書、推論をPC内だけで処理
- タスクバートレイからAI ON/OFF、文脈保持、文書分野、カスタム指示、CPU/GPU設定を変更
- AIは初期ON。トレイとAIモデルはサインイン時に起動し、不要な場合はトレイからOFFにできます
- 医学・法律・技術などの文書分野を指定し、その指示をrerankerへ渡せる
- AIが失敗・停止してもMozcの候補を使うフェイルセーフ設計

## 対応OSと必要環境

- Windows 10 22H2（build 19045）またはWindows 11、x64
- 8GB RAM以上（16GB推奨）
- 空き容量 約3GB
- CPU実行対応（INT8モデル: 67.8MB、CUDA / DirectML GPU対応: FP16 134MB）。対応GPUがあるPCでは自動的にCUDA、次にDirectMLのGPU推論を優先
- インストールには管理者権限が必要

詳細は [システム要件](docs/SYSTEM_REQUIREMENTS_JA.md) および [モデル軽量化・GPU推論技術解説](docs/DISTILLATION_QUANTIZATION_AND_GPU_INFERENCE_JA.md) を参照してください。

## インストール

1. [Releases](https://github.com/YAMA-TANA/yamatana-ai-ime/releases) から最新の `.msi` と `SHA256SUMS.txt` をダウンロードします。
2. PowerShellで `Get-FileHash .\Yamatana-AI-IME-MOZC-Ver-<version>-x64.msi -Algorithm SHA256` を実行し、公開ハッシュと一致することを確認します。配布ファイル名は既存インストールとの互換性のため、旧内部名を維持しています。
3. `Install-Yamatana-AI-IME.cmd` をダブルクリックしてインストールします。画面にMSI終了コードが表示され、`3010` または `1641` の場合は「インストール成功・PCの再起動が必要」と表示されます。MSIを直接実行する場合は、完了後のログまたは終了コードを確認してください。
4. サインアウトまたは再起動後、トレイとAIモデルが起動します。`Win + Space` で **Fumiori AI IME** を選択します。
5. 通知領域のFumioriアイコンからAIをOFFにできます。アイコンが隠れている場合は、タスクバーの `^` を開いてください。

本ソフトはMSIXではありません。既存のMozc TSF登録順序を維持したMSIで配布します。

## AI設定

トレイメニューの「設定」から、前後文脈の保持量、文書分野、任意のカスタム指示、辞書認識、CPU/GPU/自動選択を変更できます。カスタム指示には、変換判断に必要な内容だけを入力してください。設定と入力内容はローカルに保存・処理されます。

## アンインストール

Windowsの **設定 → アプリ → インストールされているアプリ → Fumiori AI IME → アンインストール** を選びます。完了後にサインアウトまたは再起動してください。

## プライバシー

アプリ本体は入力文字、文脈、辞書、AI処理結果を外部へ送信せず、テレメトリも実装していません。詳細は [PRIVACY.md](PRIVACY.md) を参照してください。GitHubからリリースやモデルを取得する操作は、GitHub側の通常のアクセスログ・プライバシーポリシーの対象です。

## 開発と再現可能性

大容量ONNXモデルはGit履歴に含めません。[model-manifest.json](model-manifest.json) に固定したリリース資産を [fetch-model.ps1](scripts/fetch-model.ps1) が取得し、SHA-256一致時のみ展開します。Mozcは [build-config.json](build-config.json) で公開forkと固定commitを指定します。Windowsワークフローはテスト、Mozc/AIランタイムのビルド、MSI生成、ハッシュ生成を自動化します。

開発参加方法は [CONTRIBUTING.md](CONTRIBUTING.md)、脆弱性報告は [SECURITY.md](SECURITY.md) を参照してください。

## ライセンス

Fumiori独自部分は [Apache License 2.0](LICENSE) です。Mozc、Ruri/ONNXモデル、辞書、同梱ライブラリには各上流ライセンスが適用されます。詳細は [NOTICE](NOTICE) と [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。

## Code signing policy

署名対象は、保護されたリリース工程でソースから生成され、GitHub Actionsの成果物として保存されたDLL / EXE / MSIだけです。現在のBetaは未署名です。方針は [CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md)、SignPath Foundation向けの申請準備・来歴一覧は [docs/SIGNPATH_READINESS.md](docs/SIGNPATH_READINESS.md) に記載しています。
