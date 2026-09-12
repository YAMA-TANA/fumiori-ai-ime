"""Shared release-version handling for the MSI and PyInstaller runtime."""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_PRODUCT_VERSION = "2.0.8.0"
DEFAULT_RELEASE_LABEL = "2.0.8-beta"
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def parse_product_version(value: str) -> tuple[int, int, int, int]:
    """Validate the numeric MSI/file version and return its four components."""
    match = _VERSION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError("product version must contain four numeric components")
    parts = tuple(int(part) for part in match.groups())
    if parts[0] > 255 or parts[1] > 255 or parts[2] > 65535 or parts[3] > 65535:
        raise ValueError("product version is outside MSI/file-version limits")
    return parts


def write_pyinstaller_version_info(
    target: str | Path,
    product_version: str,
    release_label: str,
) -> Path:
    """Write metadata whose numeric version matches the MSI upgrade version."""
    parts = parse_product_version(product_version)
    label = str(release_label).strip()
    if not label or any(char in label for char in "\r\n'"):
        raise ValueError("release label contains unsupported characters")
    dotted = ".".join(str(part) for part in parts)
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={parts!r},
    prodvers={parts!r},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '041104B0',
        [StringStruct('CompanyName', 'Yamatana'),
         StringStruct('FileDescription', 'Yamatana AI IME (MOZC Ver)'),
         StringStruct('FileVersion', '{dotted}'),
         StringStruct('InternalName', 'YamatanaAIIME'),
         StringStruct('LegalCopyright', 'Copyright (c) 2026 Yamatana'),
         StringStruct('OriginalFilename', 'YamatanaAIIME.exe'),
         StringStruct('ProductName', 'Yamatana AI IME (MOZC Ver)'),
         StringStruct('ProductVersion', '{label}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1041, 1200])])
  ]
)
""",
        encoding="utf-8",
    )
    return output
