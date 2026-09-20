# Fumiori AI IME

ローカルAIでMozcの変換候補を文脈に合わせて並べ替えるWindows日本語IMEです。標準エンジンはDual-Encoder 70Mで、候補文字列を生成せず、Mozcが提示した候補の順番だけを変更します。

## インストール後

1. Windowsへ `Fumiori AI IME` が登録されます。
2. 通知領域へFumioriのトレイアイコンが起動します。
3. 初回だけ「まず、ここだけ」という短い案内が表示されます。
4. AIは既定でONになり、サインイン時にDual-Encoder 70Mローカルモデルを読み込みます。
5. AIが不要な場合は、トレイアイコンを右クリックして「AIを使用する (ON)」を選びます。

### トレイアイコンが見つからない場合

1. タスクバー右下の `^`（隠れているインジケーターを表示）を開きます。
2. 丸い電源マークのYamatanaアイコンを右クリックします。
3. 「AIを使用する (OFF)」を選びます。チェックが付き、アイコンが緑になればONです。

初回案内は、トレイの「最初の使い方…」からいつでも再表示できます。

## 主な設定

- AIモデルをサインイン時に自動起動するか（既定ON）
- 文脈を使用するか、保持する最大文字数
- 医学・法律・ビジネス・IT・学術・創作などの文書分野
- 任意の日本語によるカスタム指示（全分野で常時利用、最大500文字）
- 辞書に基づく表記チェック
- CPU / GPU / 自動選択

入力内容と利用統計は収集せず、AI処理は端末内で完結します。

## 新しいPCへの配布

[新規PC用セットアップZIP](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/download/v2.1.1-beta/Fumiori-AI-IME-New-PC-Setup-v2.1.1-beta.zip)をダウンロードし、「すべて展開」後に `Install-Fumiori-AI-IME.cmd` をダブルクリックしてください。管理者権限を許可すると、モデル同梱のv2.1.0-beta MSIとv2.1.1-betaの更新ファイルを自動取得し、それぞれのSHA-256を照合して適用します。更新用ZIPだけを展開しても新規インストールはできません。

終了コード `3010` / `1641` の場合は「PCの再起動が必要です」と表示されます。`0` の場合もIME登録とトレイ起動を反映するため、**サインアウトまたは再起動**してください。ログは `%LOCALAPPDATA%\FumioriAIIME\Setup\v2.1.1-beta\install-msi.log` に保存されます。

未署名ベータ版のため、Windowsの発行元やSmartScreenの警告が出る場合があります。必要環境はWindows 10 22H2／11 x64、RAM8GB以上、空き容量3GB以上、管理者権限と初回ダウンロード用のインターネット接続です。AI変換時はインターネット接続を必要としません。

## 既存インストールの更新

[最新版リリース](https://github.com/YAMA-TANA/fumiori-ai-ime/releases/tag/v2.1.1-beta)の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` を展開し、`scripts\apply_hotfix.ps1` を管理者PowerShellから実行します。元の `Install-Yamatana-AI-IME.cmd` はMSIと同じ配布フォルダーに置かれている場合にのみ使用してください。

MSIX化の技術判断とMicrosoft Store向けの選択肢は、同梱の `MSIX_AND_STORE_NOTES_JA.md` を参照してください。
