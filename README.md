# Fumiori AI IME

Fumiori AI IMEは、Mozcの変換候補をローカルAI rerankerで並べ替えるWindows向け日本語IMEです。文脈、文書分野、ユーザー辞書相当の語彙情報を変換判断に使いながら、入力内容を外部へ送信しません。

> **Beta / 未署名** — 現在公開中のBetaは検証用の未署名ビルドです。Windowsの警告が表示される場合があります。SignPath FoundationによるOSSコード署名の申請準備中であり、署名済みであるかのような表示は行いません。

> **最新リリース: v2.1.1-beta** — [GitHub Releaseからダウンロード](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)できます。Dual-Encoder 70M を標準ランタイムとし、入力中の候補ベクトル先読みと10候補の再順位付けに対応しています。

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
- Dual-Encoder 70Mを標準エンジンとして、文脈ベクトルと候補ベクトルの内積で高速に再順位付け
- 入力中に文脈と候補ベクトルを先読みし、明示変換では最大10候補を1バッチで処理
- 固定候補の埋め込みを最大50万語までバックグラウンドでSQLiteに保存し、次回以降の変換で再利用
- 旧Cross-Encoder LoRAアンサンブルは既存モデルとの互換フォールバックとして利用可能
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

## 推論エンジン

v2.1.0-beta 以降の標準構成は Dual-Encoder 70M です。文脈を一度ベクトル化して候補ベクトルを比較するため、候補の追加や複数文節の変換でも推論量が増えにくく、入力中の先読みを使えます。Dual-Encoder のモデルが見つからない既存環境では、旧 Cross-Encoder LoRA アンサンブルへ自動的にフォールバックします。

## インストール

### 新しいPCに初めてインストールする場合

1. [新規PC用セットアップZIP（v2.1.1-beta）](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Fumiori-AI-IME-New-PC-Setup-v2.1.1-beta.zip)をダウンロードし、ZIPを「すべて展開」します。更新用の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` **だけでは新規インストールできません**。
2. 展開した `Install-Fumiori-AI-IME.cmd` をダブルクリックして管理者権限を許可します。PowerShellを操作したり、MSIと更新ZIPを別々にダウンロードしたりする必要はありません。インストーラーが [v2.1.0-betaのモデル同梱MSI](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.0-beta) と [v2.1.1-betaの更新ファイル](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)を取得し、両方のSHA-256を検証してインストールします。初回セットアップにはインターネット接続、空き容量3GB以上、管理者権限が必要です。
3. 完了メッセージを確認し、**サインアウトまたは再起動**します。MSIが終了コード `3010` または `1641` を返した場合は、セットアップ画面に「PCの再起動が必要です」と表示します。終了コード `0` の場合も、IME登録とトレイ起動のためサインアウトまたは再起動してください。
4. サインイン後、`Win + Space` で **Fumiori AI IME** を選択します。AIは初期ONで、通知領域のアイコンからOFFにできます。

ベータ版は未署名で、Windows SmartScreenや発行元の警告が出る場合があります。配布物とSHA-256を確認し、信頼できる場合にのみ実行してください。MSIインストールログは `%LOCALAPPDATA%\FumioriAIIME\Setup\v2.1.1-beta\install-msi.log` に残ります。インストーラー本体は [`scripts/fresh-install/`](scripts/fresh-install/) に公開しています。

### 既にインストール済みの場合

[最新版のリリース](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)から `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` を展開し、`scripts\apply_hotfix.ps1` を管理者 PowerShell で実行してください。これは既存MSIへの差分更新であり、新規PC向けセットアップではありません。

配布ファイル名やインストール先の旧内部名は既存環境との互換性のため維持しています。本ソフトはMSIXではなくMSIで配布します。

## AI設定

トレイメニューの「設定」から、前後文脈の保持量、文書分野、任意のカスタム指示、辞書認識、CPU/GPU/自動選択を変更できます。カスタム指示には、変換判断に必要な内容だけを入力してください。設定と入力内容はローカルに保存・処理されます。

## アンインストール

Windowsの **設定 → アプリ → インストールされているアプリ → Fumiori AI IME → アンインストール** を選びます。完了後にサインアウトまたは再起動してください。

## プライバシー

アプリ本体は入力文字、文脈、辞書、AI処理結果を外部へ送信せず、テレメトリも実装していません。詳細は [PRIVACY.md](PRIVACY.md) を参照してください。GitHubからリリースやモデルを取得する操作は、GitHub側の通常のアクセスログ・プライバシーポリシーの対象です。

## 開発と再現可能性

大容量ONNXモデルはGit履歴に含めません。[model-manifest.json](model-manifest.json) に固定したDual-Encoder標準モデルと互換フォールバックモデルのリリース資産を [fetch-model.ps1](scripts/fetch-model.ps1) が取得し、SHA-256一致時のみ展開します。Mozcは [build-config.json](build-config.json) で公開forkと固定commitを指定します。Windowsワークフローはテスト、Mozc/AIランタイムのビルド、MSI生成、ハッシュ生成を自動化します。

開発参加方法は [CONTRIBUTING.md](CONTRIBUTING.md)、脆弱性報告は [SECURITY.md](SECURITY.md) を参照してください。

## ライセンス

Fumiori独自部分は [Apache License 2.0](LICENSE) です。Mozc、Ruri/ONNXモデル、辞書、同梱ライブラリには各上流ライセンスが適用されます。詳細は [NOTICE](NOTICE) と [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。

## Code signing policy

署名対象は、保護されたリリース工程でソースから生成され、GitHub Actionsの成果物として保存されたDLL / EXE / MSIだけです。現在のBetaは未署名です。方針は [CODE_SIGNING_POLICY.md](CODE_SIGNING_POLICY.md)、SignPath Foundation向けの申請準備・来歴一覧は [docs/SIGNPATH_READINESS.md](docs/SIGNPATH_READINESS.md) に記載しています。
