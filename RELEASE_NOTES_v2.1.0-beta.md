# Fumiori AI IME v2.1.0-beta (Dual-Encoder 70M Architecture)

次世代推論エンジン **Dual-Encoder（内積検索型）AI Reranker** を搭載したメジャーアップデートです。
従来の Cross-Encoder に比べ推論遅延を 95% 削減（平均 9.1ms、内積単体 0.23ms）し、タイピング打鍵に完全追従する極超低遅延と高い変換精度を両立しました。

---

## 主な変更点

### 1. Dual-Encoder（内積検索型）推論アーキテクチャの導入
- **95% の推論高速化**: 文脈エンコードと事前キャッシュ候補ベクトルによる行列積（Dot-Product）検索により、推論遅延を従来の 75〜280ms から **平均 9.10ms（内積単体 0.23ms / 230マイクロ秒）** へ劇的に短縮しました。
- **低リソース動作**: 常駐メモリ 280MB 以下で動作し、バックグラウンドでの軽快な打鍵感を保証します。

### 2. 異字同訓 Hard-Negative 対照学習
- 「映画を映す／ノートに写す」「部屋を空ける／窓を開ける」「罪を犯す／危険を冒す」「税金を納める／国を治める」「冬の温かいスープ／春の暖かい日差し」など、文脈共起が酷似する微細な同訓異字を Hard Negative として ModernBERT-70M に対照学習（InfoNCE）。意味境界を高次元ベクトル空間でシャープに分離しました。

### 3. 句読点の「20〜30文字安全バウンダリ」
- 直前20文字以内の読点（`、`）をソフトポーズとして認識し、文脈切断を防止。「〜において、決裁された」のような修飾節・主語の重要文脈が消失する問題を根本解決しました。

### 4. 複数文節一括変換における「全内積（Batch Dot-Product）」
- 3〜4文節をまとめた一括変換キー押下時、文節数（$S$）の文脈を一括フォワードし、全候補行列と一括全内積（All-Dot-Product）を計算。Cross-Encoder で 400〜520ms かかっていた複数文節処理を **17〜64ms（内積計算は 0.3〜1.5ms）** で瞬時に完了します。

### 5. コンテンツシグナル安全ゲート（ゼロ・曖昧文脈での珍奇語防止）
- 文脈が空の場合や、指示代名詞（「これの」「その」）のみで意味決定が困難な場合、無理な内積推論を行わず Mozc の辞書頻度1位（例: 「こうせい」➔「構成」）を確実に維持。日常使用頻度の低い珍奇語（「江青」など）への誤爆を 100% 根絶しました。

### 6. 実務特化500問ストレステストの完遂
- 実Mozc候補、数詞含有（「第3四半期」「残り2日」等）、奇妙な変換単位（助詞活用巻き込み）、IT略語（Slack, AWS, MTG, 情シス, リスケ等）を網羅した実務500問テストにおいて、**Mozc素順位 61.8% ➔ AI適用後 81.0%（+19.2%純増、Mozc誤り 139件を救済）** を達成しました。

---

## 配布物

- `Yamatana-AI-IME-MOZC-Ver-2.1.0-beta-x64.msi`
- `Yamatana-AI-IME-v2.1.0-beta-hotfix-20260919-r6.zip`（推奨。非同期プリフェッチ＋Mozcプロセスのロック解除待ち／リトライ対応）
- `SHA256SUMS.txt`

## Beta notice

これは未署名のベータ版MSIです。インストール時にWindowsのSmartScreenまたは発行元に関する警告が表示される場合があります。

---

## 2026-09-19 production hotfix

- RealtimeDecoder の内部リクエストが `request_type=CONVERSION` のまま dispatch されるため、AI rewriter の capability を `CONVERSION | PREDICTION` に修正しました。入力中の context prefetch が実際の Mozc 経路で呼ばれます。
- 入力中の文脈ベクトル先読み、Space 時の候補内積、文脈変更時のキャッシュ破棄を維持しています。
- prefetch を名前付きパイプの受付処理から切り離し、最新の入力だけをバックグラウンドで処理します。Space 時はキャッシュ参照と内積計算を優先します。
- r6 では `mozc_broker`、`mozc_server`、`mozc_renderer` をコピー試行ごとに強制終了し、再生成されても再試行で追従します。常駐監視で再生成される `mozc_cache_service` は対象外です。

### Existing installation

既存 MSI を使っている場合は、次の hotfix ZIP を展開し、正規パスまたは互換パスのスクリプトを管理者 PowerShell から実行してください。

- 推奨: `Yamatana-AI-IME-v2.1.0-beta-hotfix-20260919-r6.zip`
- SHA-256: `354768E338D721CCA370E317C9E04E0075EC767551F0438C1AAD607E91DAAE98`
- 正規スクリプト: `scripts\apply_hotfix.ps1`
- 互換スクリプト: `scripts\apply\_hotfix.ps1`
- 旧 20260918 / 20260919 / r2 / r3 / r4 / r5 ZIP は使用しないでください。

実行コマンド:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\lotof\Videos\research\live2d\yamatana-ai-ime\build\hotfix-20260919-r6\scripts\apply_hotfix.ps1"
```

この hotfix は現在の MSI と同じランタイムに対する差分配布です。完全な MSI の再生成は WiX の巨大 CAB bind 工程が停止するため、今回の Release では hotfix を追加配布しています。
