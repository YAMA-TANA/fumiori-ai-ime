"""Update the published site and install docs for the verified new-PC bundle.

Run once when shipping a combined installer ZIP. No binary/branding renames.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/YAMA-TANA/fumiori-ai-ime/releases'
ASSET = BASE + '/download/v2.1.1-beta/Fumiori-AI-IME-New-PC-Setup-v2.1.1-beta.zip'
LATEST = BASE + '/tag/v2.1.1-beta'


def replace_once(text: str, old: str, new: str, path: str) -> str:
    found = text.count(old)
    if found != 1:
        raise AssertionError(f'{path}: expected one target, got {found}: {old[:95]!r}')
    return text.replace(old, new)


def put(path: str, before: str, after: str) -> None:
    if before == after:
        raise AssertionError(f'{path}: no changes')
    (ROOT / path).write_text(after, encoding='utf-8', newline='\n')
    print('Updated', path)


path = 'docs/index.html'
s = (ROOT / path).read_text(encoding='utf-8')
old_release = BASE + '/tag/v2.0.0-beta'
assert s.count(old_release) == 3, s.count(old_release)
s = s.replace(old_release, ASSET)
s = replace_once(s, 'Windows 10 / 11 · v2.0.0-beta',
                 'Windows 10 / 11 · v2.1.1-beta', path)
s = replace_once(s, '>Windows版を試す <span aria-hidden="true">→</span></a>',
                 '>新規PC用セットアップを入手 <span aria-hidden="true">→</span></a>', path)
s = replace_once(s, '<h3>AIは初期OFF</h3><p>通常のMozc変換はそのまま使えます。必要な時だけトレイからAIモデルを読み込みます。</p><span class="arch-tag">ON DEMAND</span>',
                 '<h3>AIは初期ON・切替可能</h3><p>サインイン時にAIを起動します。不要な場合は通知領域からOFFにできます。</p><span class="arch-tag">LOCAL / OPTIONAL</span>', path)
start = s.index('    <section class="section install-section" id="start">')
end = s.index('    <section class="cta-section">', start)
s = s[:start] + '''    <section class="section install-section" id="start">
      <div class="shell install-layout">
        <div class="section-heading reveal"><span class="kicker">GET STARTED</span><h2>新しいWindows PCにも、<br>まとめてインストール。</h2><p>Windows 10/11 x64向けの未署名ベータ版です。セットアップ時は管理者権限とインターネット接続、空き容量3GB以上が必要です。入力内容やAI推論はPC内で処理されます。</p><a class="button button-primary" href="''' + ASSET + '''">新規PC用セットアップZIPを入手 →</a><p><small>既にインストール済みの方は<a href="''' + LATEST + '''">v2.1.1-beta更新用ZIP</a>をご利用ください。更新用ZIP単体では新規インストールできません。</small></p></div>
        <ol class="install-steps reveal">
          <li><span>01</span><div><strong>ZIPを「すべて展開」</strong><p>新規PC用セットアップZIPを展開し、<code>Install-Fumiori-AI-IME.cmd</code>をダブルクリックします。</p></div></li>
          <li><span>02</span><div><strong>管理者権限を許可</strong><p>正規のMSI（AIモデル同梱）と最新版の更新を自動取得し、SHA-256を検証して適用します。ベータ版は未署名のためWindowsの警告が出る場合があります。</p></div></li>
          <li><span>03</span><div><strong>サインアウトまたは再起動</strong><p>セットアップの終了表示を確認し、Windowsに再サインインしてから<code>Win + Space</code>でFumiori AI IMEを選びます。再起動が必須の場合は明示します。</p></div></li>
        </ol>
      </div>
    </section>

''' + s[end:]
s = replace_once(s, '>Windows版を試す →</a>', '>新規PC用セットアップを入手 →</a>', path)
assert 'releases/tag/v2.0.0-beta' not in s
put(path, (ROOT / path).read_text(encoding='utf-8'), s)

path = 'README.md'
s = (ROOT / path).read_text(encoding='utf-8')
start = s.index('## インストール\n')
end = s.index('## AI設定\n', start)
s = s[:start] + '''## インストール

### 新しいPCに初めてインストールする場合

1. [新規PC用セットアップZIP（v2.1.1-beta）](''' + ASSET + ''')をダウンロードし、ZIPを「すべて展開」します。更新用の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` **だけでは新規インストールできません**。
2. 展開した `Install-Fumiori-AI-IME.cmd` をダブルクリックして管理者権限を許可します。PowerShellを操作したり、MSIと更新ZIPを別々にダウンロードしたりする必要はありません。インストーラーが [v2.1.0-betaのモデル同梱MSI](''' + BASE + '''/tag/v2.1.0-beta) と [v2.1.1-betaの更新ファイル](''' + LATEST + ''')を取得し、両方のSHA-256を検証してインストールします。初回セットアップにはインターネット接続、空き容量3GB以上、管理者権限が必要です。
3. 完了メッセージを確認し、**サインアウトまたは再起動**します。MSIが終了コード `3010` または `1641` を返した場合は、セットアップ画面に「PCの再起動が必要です」と表示します。終了コード `0` の場合も、IME登録とトレイ起動のためサインアウトまたは再起動してください。
4. サインイン後、`Win + Space` で **Fumiori AI IME** を選択します。AIは初期ONで、通知領域のアイコンからOFFにできます。

ベータ版は未署名で、Windows SmartScreenや発行元の警告が出る場合があります。配布物とSHA-256を確認し、信頼できる場合にのみ実行してください。MSIインストールログは `%LOCALAPPDATA%\\FumioriAIIME\\Setup\\v2.1.1-beta\\install-msi.log` に残ります。インストーラー本体は [`scripts/fresh-install/`](scripts/fresh-install/) に公開しています。

### 既にインストール済みの場合

[最新版のリリース](''' + LATEST + ''')から `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` を展開し、`scripts\\apply_hotfix.ps1` を管理者 PowerShell で実行してください。これは既存MSIへの差分更新であり、新規PC向けセットアップではありません。

配布ファイル名やインストール先の旧内部名は既存環境との互換性のため維持しています。本ソフトはMSIXではなくMSIで配布します。

''' + s[end:]
put(path, (ROOT / path).read_text(encoding='utf-8'), s)

path = 'docs/README_DISTRIBUTION_JA.md'
s = (ROOT / path).read_text(encoding='utf-8')
start = s.index('## 配布ファイル\n')
s = s[:start] + '''## 新しいPCへの配布

[新規PC用セットアップZIP](''' + ASSET + ''')をダウンロードし、「すべて展開」後に `Install-Fumiori-AI-IME.cmd` をダブルクリックしてください。管理者権限を許可すると、モデル同梱のv2.1.0-beta MSIとv2.1.1-betaの更新ファイルを自動取得し、それぞれのSHA-256を照合して適用します。更新用ZIPだけを展開しても新規インストールはできません。

終了コード `3010` / `1641` の場合は「PCの再起動が必要です」と表示されます。`0` の場合もIME登録とトレイ起動を反映するため、**サインアウトまたは再起動**してください。ログは `%LOCALAPPDATA%\\FumioriAIIME\\Setup\\v2.1.1-beta\\install-msi.log` に保存されます。

未署名ベータ版のため、Windowsの発行元やSmartScreenの警告が出る場合があります。必要環境はWindows 10 22H2／11 x64、RAM8GB以上、空き容量3GB以上、管理者権限と初回ダウンロード用のインターネット接続です。AI変換時はインターネット接続を必要としません。

## 既存インストールの更新

[最新版リリース](''' + LATEST + ''')の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` を展開し、`scripts\\apply_hotfix.ps1` を管理者PowerShellから実行します。元の `Install-Yamatana-AI-IME.cmd` はMSIと同じ配布フォルダーに置かれている場合にのみ使用してください。

MSIX化の技術判断とMicrosoft Store向けの選択肢は、同梱の `MSIX_AND_STORE_NOTES_JA.md` を参照してください。
'''
put(path, (ROOT / path).read_text(encoding='utf-8'), s)

path = 'RELEASE_NOTES_v2.1.1-beta.md'
s = (ROOT / path).read_text(encoding='utf-8')
s += '''\n## 新しいPC用セットアップパック\n\n- [新規PC用セットアップZIP](''' + ASSET + ''')には `Install-Fumiori-AI-IME.cmd` と `install-fresh.ps1` が入ります。展開してCMDをダブルクリックすると、SHA-256照合のうえモデル同梱v2.1.0-beta MSIと本リリースの差分更新を順番に適用します。\n- 既存の `Yamatana-AI-IME-v2.1.1-beta-candidate10.zip` は差分更新専用であり、それだけでは新規インストールできません。\n- 変更の反映にはサインアウトまたは再起動が必要です。MSIが再起動必須を返した場合は明示します。\n'''
put(path, (ROOT / path).read_text(encoding='utf-8'), s)
