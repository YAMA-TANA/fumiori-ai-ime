#!/usr/bin/env python3
"""Update URLs and project-relative links after the repository rename.

Historical installer filenames, product identifiers and local filesystem paths
must not be renamed: those may still be required to install old releases.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = "fumiori-ai-ime"
BASE = f"https://yama-tana.github.io/{NEW}/"
SELF = Path(__file__).resolve()


def rewrite(text: str) -> str:
    for host in (
        r"https?://(?:www\.)?github\.com/YAMA-TANA/",
        r"https?://api\.github\.com/repos/YAMA-TANA/",
        r"https?://raw\.githubusercontent\.com/YAMA-TANA/",
        r"https?://yama-tana\.github\.io/",
    ):
        text = re.sub(
            rf"(?i)({host})yamatana-ai-ime(?=[/#?\s\"'<>)]|$)",
            rf"\g<1>{NEW}",
            text,
        )
    # Website-root references, not local directories or historical artifact names.
    text = text.replace("/yamatana-ai-ime/", f"/{NEW}/")
    text = re.sub(
        r"(?<=\")/yamatana-ai-ime(?=[\"?#])|(?<=')/yamatana-ai-ime(?=['?#])",
        f"/{NEW}",
        text,
    )
    return text


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    updated: list[str] = []
    for raw in tracked:
        if not raw:
            continue
        path = ROOT / raw.decode("utf-8")
        if path.resolve() == SELF or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            original = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        changed = rewrite(original)
        if changed != original:
            path.write_bytes(changed.encode("utf-8"))
            updated.append(path.relative_to(ROOT).as_posix())

    # Make search indexing refer to the new site rather than the old project URL.
    for page_name in ("index.html", "oss.html", "oss-en.html"):
        path = ROOT / "docs" / page_name
        text = path.read_text(encoding="utf-8")
        original = text
        canonical = BASE if page_name == "index.html" else BASE + page_name
        if 'rel="canonical"' not in text:
            text = text.replace("</head>", f'  <link rel="canonical" href="{canonical}">\n</head>', 1)
        if page_name == "index.html" and 'property="og:url"' not in text:
            text = text.replace("</head>", f'  <meta property="og:url" content="{BASE}">\n</head>', 1)
        if text != original:
            path.write_text(text, encoding="utf-8")
            relative = path.relative_to(ROOT).as_posix()
            if relative not in updated:
                updated.append(relative)

    remaining: list[str] = []
    for raw in tracked:
        if not raw:
            continue
        path = ROOT / raw.decode("utf-8")
        if path.resolve() == SELF or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        if any(fragment in text for fragment in (
            "github.com/yama-tana/yamatana-ai-ime",
            "github.io/yamatana-ai-ime",
            "/yamatana-ai-ime/",
        )):
            remaining.append(path.relative_to(ROOT).as_posix())
    if remaining:
        raise SystemExit("Old URL references remain: " + ", ".join(remaining))
    for asset in ("index.html", "styles.css", "app.js", "promo.css", "policy.css", "oss.html", "oss-en.html"):
        if not (ROOT / "docs" / asset).is_file():
            raise SystemExit(f"Missing published asset: docs/{asset}")
    print(f"Updated {len(updated)} tracked files: {', '.join(updated) if updated else 'none'}")
    print("Verified: no old URLs and all key Pages assets exist")


if __name__ == "__main__":
    main()
