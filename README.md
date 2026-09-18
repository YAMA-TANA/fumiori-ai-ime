# Fumiori AI IME

Fumiori AI IMEは、Mozcの変換候補をローカルAI rerankerで並べ替えるWindows向け日本語IMEです。文脈、文書分野、ユーザー辞書相当の語彙情報を変換判断に使いながら、入力内容を外部へ送信しません。

> **Beta / 未署名** — 現在公開中のBetaは検証用の未署名ビルドです。Windowsの警告が表示される場合があります。SignPath FoundationによるOSSコード署名の申請準備中であり、署名済みであるかのような表示は行いません。

> **最新リリース: v2.0.10-beta** — [GitHub Releaseからダウンロード](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.0.10-beta)できます。Fumiori AI IMEの表示名、ドット絵ゲーム風候補ウィンドウ、`CHAIN +1` の連鎖演出を含むベータ版です。

## Fumioriの変換体験

変換候補ウィンドウは、8bitゲームを思わせる限定色とステップ角のカードUIで表示します。AIが文脈に合う候補を選んだときは `CHAIN +1` を表示し、変換を楽しく確認できるようにしています。見た目はゲーム風でも、Mozcの通常変換とAI失敗時のフェイルセーフは維持しています。

## 特徴

- Mozcベースの通常変換を保ったまま、AI有効時だけ候補を再順位付け
- 70M級Ruri v3 student rerankerをIME向けに蒸留し、全候補を1バッチでONNX Runtime実行
- 入力、前後文脈、カスタム指示、辞書、推論をPC内だけで処理
- タスクバートレイからAI ON/OFF、文脈保持、文書分野、カスタム指示、CPU/GPU設定を変更
- AIは初期OFF。トレイは状態切替と設定のため起動しますが、OFF時はAIモデルをロードしません
- 医学・法律・技術などの文書分野を指定し、その指示をrerankerへ渡せる
- AIが失敗・停止してもMozcの候補を使うフェイルセーフ設計

## 対応OSと必要環境

- Windows 10 22H2（build 19045）またはWindows 11、x64
- 8GB RAM以上（16GB推奨）
- 空き容量 約3GB
- CPU実行対応（INT8モデル: 67.8MB、DirectML GPU対応: FP16 134MB）。対応GPUがあるPCでは自動的にDirectML GPU推論を優先
- インストールには管理者権限が必要

詳細は [システム要件](docs/SYSTEM_REQUIREMENTS_JA.md) および [モデル軽量化・GPU推論技術解説](docs/DISTILLATION_QUANTIZATION_AND_GPU_INFERENCE_JA.md) を参照してください。

## インストール

1. [v2.0.10-betaのReleaseページ](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.0.10-beta)から最新の `.msi` と `SHA256SUMS.txt` をダウンロードします。
2. PowerShellで `Get-FileHash .\<ダウンロードしたMSIファイル名> -Algorithm SHA256` を実行し、公開ハッシュと一致することを確認します。既存インストールとの互換性のため、配布ファイル名だけは旧内部名を維持しています。
3. MSIをダブルクリックし、プライバシー説明を確認してインストールします。
4. サインアウトまたは再起動後、`Win + Space` で **Fumiori AI IME** を選択します。
5. 通知領域のFumioriアイコンを開き、必要なときだけ **AIをON** にします。初期状態はOFFです。アイコンが隠れている場合は、タスクバーの `^` を開いてください。

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
