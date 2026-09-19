# Yamatana AI IME v2.0.8-beta

## インストール結果の表示

- `install-msi.ps1` と `Install-Yamatana-AI-IME.cmd` を配布物へ同梱。
- MSI終了コード `3010` / `1641` を「インストール成功・PCの再起動が必要」と日本語で表示。
- MSIログを `Yamatana-AI-IME-install.log` に保存し、失敗時も終了コードを表示。

## 使い方

MSIと同じフォルダーにある `Install-Yamatana-AI-IME.cmd` を実行してください。
`3010` または `1641` が表示された場合、インストールは成功していますが、変更を完全に反映するためPCの再起動が必要です。
