"""Build the standalone Yamatana AI IME (MOZC Ver) MSI with WiX 4."""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
import shutil
import subprocess

try:
    from scripts.release_version import DEFAULT_PRODUCT_VERSION, DEFAULT_RELEASE_LABEL
except ModuleNotFoundError:  # Direct execution: python scripts/build_ai_msi.py
    from release_version import DEFAULT_PRODUCT_VERSION, DEFAULT_RELEASE_LABEL


ROOT = Path(__file__).resolve().parents[1]
MOZC = Path(
    os.environ.get("YAMATANA_MOZC_ROOT", str(ROOT / "build" / "mozc-src"))
).resolve() / "src"
RUNTIME = ROOT / "dist" / "YamatanaAIIME"
BUILD_DIR = ROOT / "build" / "distribution-msi"
RELEASE_DIR = ROOT / "release"
PRODUCT_VERSION = os.environ.get("YAMATANA_PRODUCT_VERSION", DEFAULT_PRODUCT_VERSION)
RELEASE_LABEL = os.environ.get("YAMATANA_RELEASE_LABEL", DEFAULT_RELEASE_LABEL)
MSI_OUT = RELEASE_DIR / f"Yamatana-AI-IME-MOZC-Ver-{RELEASE_LABEL}-x64.msi"
UPGRADE_CODE = "A9FD6996-83DE-4DBE-9BE9-8C7F9016493A"


def _stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def _find_bazel_output(relative: str) -> Path:
    matches = sorted(
        MOZC.glob(f"bazel-out/*/bin/{relative}"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"Bazel output not found: {relative}")
    return matches[0].resolve()


def _make_privacy_rtf(source: Path, target: Path) -> None:
    def encode(text: str) -> str:
        result: list[str] = []
        for char in text.replace("\r\n", "\n"):
            code = ord(char)
            if char in "\\{}":
                result.append("\\" + char)
            elif char == "\n":
                result.append("\\par\n")
            elif 32 <= code < 127:
                result.append(char)
            else:
                if code > 32767:
                    code -= 65536
                result.append(f"\\u{code}?")
        return "".join(result)

    body = encode(source.read_text(encoding="utf-8"))
    target.write_text(
        "{\\rtf1\\ansi\\deff0\\uc1"
        "{\\fonttbl{\\f0\\fnil\\fcharset128 Yu Gothic UI;}}"
        "\\viewkind4\\pard\\f0\\fs20\n" + body + "\n}",
        encoding="ascii",
    )


def _make_runtime_fragment(target: Path) -> int:
    files = sorted(path for path in RUNTIME.rglob("*") if path.is_file())
    directory_set: set[str] = set()
    for file_path in files:
        parent = file_path.parent.relative_to(RUNTIME)
        while parent != Path("."):
            directory_set.add(parent.as_posix())
            parent = parent.parent
    directories = sorted(
        directory_set,
        key=lambda value: (value.count("/"), value.casefold()),
    )
    directory_ids = {
        relative: _stable_id("RuntimeDir_", relative) for relative in directories
    }

    children: dict[str, list[str]] = {}
    for relative in directories:
        parent = str(Path(relative).parent).replace("\\", "/")
        if parent == ".":
            parent = ""
        children.setdefault(parent, []).append(relative)

    def emit_directories(parent: str, indent: str) -> list[str]:
        lines: list[str] = []
        for relative in sorted(children.get(parent, []), key=str.casefold):
            name = Path(relative).name
            lines.append(
                f'{indent}<Directory Id="{directory_ids[relative]}" '
                f'Name="{html.escape(name, quote=True)}">'
            )
            lines.extend(emit_directories(relative, indent + "  "))
            lines.append(f"{indent}</Directory>")
        return lines

    xml = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">',
        "  <Fragment>",
        '    <DirectoryRef Id="YamatanaRuntimeDir">',
        *emit_directories("", "      "),
        "    </DirectoryRef>",
        "  </Fragment>",
        "  <Fragment>",
        '    <ComponentGroup Id="YamatanaRuntimeFiles">',
    ]
    component_rows: list[str] = []
    for file_path in files:
        relative = file_path.relative_to(RUNTIME).as_posix()
        component_id = _stable_id("RuntimeComponent_", relative)
        file_id = (
            "YamatanaAIIME.exe"
            if relative == "YamatanaAIIME.exe"
            else _stable_id("RuntimeFile_", relative)
        )
        parent = str(Path(relative).parent).replace("\\", "/")
        directory_id = "YamatanaRuntimeDir" if parent == "." else directory_ids[parent]
        xml.append(f'      <ComponentRef Id="{component_id}" />')
        component_rows.extend([
            f'    <Component Id="{component_id}" Directory="{directory_id}" Guid="*">',
            f'      <File Id="{file_id}" Source="{html.escape(str(file_path.resolve()), quote=True)}" KeyPath="yes" />',
            "    </Component>",
        ])
    xml.extend([
        "    </ComponentGroup>",
        "  </Fragment>",
        "  <Fragment>",
        *component_rows,
        "  </Fragment>",
        "</Wix>",
        "",
    ])
    target.write_text("\n".join(xml), encoding="utf-8")
    return len(files)


def _stage_support_files() -> Path:
    qt = MOZC / "third_party" / "qt"
    extracted = ROOT / "build" / "extracted_mozc" / "PFiles" / "Mozc"
    support = BUILD_DIR / "support"
    (support / "bin").mkdir(parents=True, exist_ok=True)
    (support / "plugins" / "platforms").mkdir(parents=True, exist_ok=True)

    for name in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
        source = qt / "bin" / name
        if not source.exists():
            source = extracted / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, support / "bin" / name)

    redist_candidates: list[Path] = []
    if os.environ.get("VCToolsRedistDir"):
        redist_candidates.append(Path(os.environ["VCToolsRedistDir"]))
    redist_candidates.extend(Path(path) for path in (
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Redist\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC",
    ) if Path(path).exists())
    crt_dirs: list[Path] = []
    for candidate in redist_candidates:
        crt_dirs.extend(candidate.glob("**/x64/Microsoft.VC*.CRT"))
        if candidate.name.startswith("Microsoft.VC"):
            crt_dirs.append(candidate)
    if not crt_dirs:
        raise FileNotFoundError("Visual C++ x64 redistributable directory not found")
    crt_dir = sorted(crt_dirs, key=lambda value: str(value), reverse=True)[0]
    for name in (
        "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
        "vcruntime140.dll", "vcruntime140_1.dll",
    ):
        source = crt_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, support / name)

    qwindows = qt / "plugins" / "platforms" / "qwindows.dll"
    if not qwindows.exists():
        qwindows = extracted / "qwindows.dll"
    if not qwindows.exists():
        qwindows = extracted / "platforms" / "qwindows.dll"
    if not qwindows.exists():
        raise FileNotFoundError(qwindows)
    shutil.copy2(qwindows, support / "plugins" / "platforms" / "qwindows.dll")
    return support


def build_msi() -> Path:
    if not RUNTIME.exists():
        raise FileNotFoundError(
            f"Distribution runtime is missing: {RUNTIME}. Build ai_ime_tray.spec first."
        )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    privacy_rtf = BUILD_DIR / "privacy_policy_ja.rtf"
    runtime_wxs = BUILD_DIR / "runtime_files.wxs"
    installer_wxs = BUILD_DIR / "installer_oss_64bit.wxs"
    _make_privacy_rtf(ROOT / "PRIVACY.md", privacy_rtf)
    file_count = _make_runtime_fragment(runtime_wxs)
    support = _stage_support_files()
    installer_text = (MOZC / "win32" / "installer" / "installer_oss_64bit.wxs").read_text(
        encoding="utf-8"
    )
    installer_text = installer_text.replace(
        '    <CustomAction Id="LaunchYamatanaTray" FileRef="YamatanaAIIME.exe" ExeCommand="--from-installer" Execute="immediate" Impersonate="yes" Return="asyncNoWait" />\n',
        "",
    ).replace(
        '      <Custom Action="LaunchYamatanaTray" Before="InstallFinalize" Condition="(NOT Installed) AND (ACTION=&quot;INSTALL&quot;)" />\n',
        "",
    )
    installer_text = installer_text.replace(
        '    <WixVariable Id="WixUILicenseRtf" Value="$(var.PrivacyRtf)" />\n',
        '    <WixVariable Id="WixUILicenseRtf" Value="$(var.PrivacyRtf)" />\n'
        '    <Property Id="WIXUI_EXITDIALOGOPTIONALTEXT" '
        'Value="インストールは完了しました。変更を反映するには Windows の再起動が必要です。" />\n',
    )
    installer_wxs.write_text(installer_text, encoding="utf-8")

    wix_command = shutil.which("wix") or shutil.which("wix.exe")
    if not wix_command:
        raise FileNotFoundError("wix.exe was not found on PATH")
    wix_exe = Path(wix_command)
    env = os.environ.copy()

    inputs = {
        "MozcVersion": PRODUCT_VERSION,
        "UpgradeCode": UPGRADE_CODE,
        "VSConfigurationName": "Release",
        "ReleaseRedistCrt64Dir": str(support),
        "AddRemoveProgramIconPath": str(MOZC / "data" / "images" / "win" / "product_icon.ico"),
        "MozcTIP32Path": str(_find_bazel_output("win32/tip/mozc_tip32.dll")),
        "MozcTIP64Path": str(_find_bazel_output("win32/tip/mozc_tip64.dll")),
        "MozcBroker64Path": str(_find_bazel_output("win32/broker/mozc_broker_main.exe")),
        "MozcServer64Path": str(_find_bazel_output("server/mozc_server_win.exe")),
        "MozcCacheService64Path": str(_find_bazel_output("win32/cache_service/mozc_cache_service.exe")),
        "MozcRenderer64Path": str(_find_bazel_output("renderer/win32/win32_renderer_main.exe")),
        "MozcToolPath": str(_find_bazel_output("gui/tool/mozc_tool_win.exe")),
        "CustomActions64Path": str(_find_bazel_output("win32/custom_action/custom_action.dll")),
        "DocumentsDir": str(MOZC / "data" / "installer"),
        "QtDir": str(support),
        "QtVer": "6",
        "PrivacyRtf": str(privacy_rtf),
    }
    command = [
        str(wix_exe), "build", "-nologo", "-arch", "x64",
        "-ext", "WixToolset.UI.wixext",
    ]
    for name, value in inputs.items():
        command.extend(["-define", f"{name}={value}"])
    command.extend([
        "-out", str(MSI_OUT),
        str(installer_wxs),
        str(runtime_wxs),
    ])
    print(f"Packaging {file_count} runtime files with Mozc and installer actions...")
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"WiX build failed with exit code {completed.returncode}")
    print(f"Built: {MSI_OUT} ({MSI_OUT.stat().st_size / 1024 / 1024:.1f} MiB)")
    return MSI_OUT


if __name__ == "__main__":
    build_msi()
