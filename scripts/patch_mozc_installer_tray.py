from __future__ import annotations

import argparse
from pathlib import Path


def remove_once(text: str, needle: str, label: str) -> str:
    if needle not in text:
        raise RuntimeError(f"unexpected Mozc installer source while patching {label}")
    return text.replace(needle, "", 1)


def patch_checkout(checkout: Path) -> None:
    installer = checkout / "src/win32/installer/installer_oss_64bit.wxs"
    text = installer.read_text(encoding="utf-8")
    text = remove_once(
        text,
        '      <CustomAction Id="LaunchYamatanaTray" FileRef="YamatanaAIIME.exe" '
        'ExeCommand="--from-installer" Execute="immediate" Impersonate="yes" '
        'Return="asyncNoWait" />\n',
        "tray custom action declaration",
    )
    text = remove_once(
        text,
        '        <Custom Action="LaunchYamatanaTray" Before="InstallFinalize" '
        'Condition="(ACTION=&quot;INSTALL&quot;)" />\n',
        "tray custom action sequence",
    )
    installer.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    args = parser.parse_args()
    patch_checkout(args.checkout.resolve())


if __name__ == "__main__":
    main()
