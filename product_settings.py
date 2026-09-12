"""Local-only product settings for Yamatana AI IME (MOZC Ver)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


PRODUCT_NAME = "Yamatana AI IME (MOZC Ver)"
PRODUCT_VERSION = "2.0.7-beta"
SETTINGS_SCHEMA = 1
LEGACY_AUTOSTART_VALUE_NAME = "Yamatana-AI-IME"
WINDOWS_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

DOMAIN_PRESETS = {
    "general": ("一般", "自然で一般的な日本語として、文脈に合う表記を優先する。"),
    "medical": ("医学・医療", "医学・医療文書として、疾患名、解剖、薬剤、治療に適切な専門表記を優先する。"),
    "legal": ("法律・行政", "法律・行政文書として、法令、契約、制度に適切な正式表記を優先する。"),
    "business": ("ビジネス", "ビジネス文書として、簡潔で正式な表記と一般的な業務用語を優先する。"),
    "software": ("IT・ソフトウェア", "IT・ソフトウェア文書として、技術用語、製品名、コード周辺の表記を優先する。"),
    "academic": ("学術・研究", "学術文書として、専門用語の一貫性と論文調の正式表記を優先する。"),
    "creative": ("創作・会話", "創作・会話文として、人物の語調と自然な口語表現を優先する。"),
    "custom": ("カスタム", ""),
}

COMPUTE_MODES = {
    "auto": "自動選択（GPU優先、利用できなければCPU）",
    "gpu": "GPU（高速、対応GPUが必要）",
    "cpu": "CPU（互換性優先）",
}

CONTEXT_LENGTHS = (0, 32, 64, 128, 256, 512)

DEFAULT_SETTINGS: dict[str, Any] = {
    "schema": SETTINGS_SCHEMA,
    # The tray and the model start with Windows by default.  Users can turn
    # the model off from the tray when they prefer ordinary Mozc conversion.
    "ai_autostart": True,
    "context_enabled": True,
    "context_chars": 128,
    "document_domain": "general",
    "custom_instruction": "",
    "lexical_grounding": True,
    "compute_mode": "auto",
}


def product_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
    if base:
        return Path(base) / "YamatanaAIIME"
    return Path(tempfile.gettempdir()) / "YamatanaAIIME"


def default_settings_path() -> Path:
    return product_data_dir() / "settings.json"


def _as_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def normalize_settings(raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    source = dict(raw or {})
    result = dict(DEFAULT_SETTINGS)
    result["ai_autostart"] = _as_bool(
        source.get("ai_autostart"), DEFAULT_SETTINGS["ai_autostart"]
    )
    result["context_enabled"] = _as_bool(
        source.get("context_enabled"), DEFAULT_SETTINGS["context_enabled"]
    )
    try:
        context_chars = int(source.get("context_chars", DEFAULT_SETTINGS["context_chars"]))
    except (TypeError, ValueError):
        context_chars = int(DEFAULT_SETTINGS["context_chars"])
    result["context_chars"] = min(CONTEXT_LENGTHS, key=lambda value: abs(value - context_chars))

    domain = str(source.get("document_domain", DEFAULT_SETTINGS["document_domain"]))
    result["document_domain"] = domain if domain in DOMAIN_PRESETS else "general"
    instruction = str(source.get("custom_instruction", "")).strip()
    # A local instruction is included in every scoring query.  Keep it short
    # enough to preserve the model's token budget and IME latency.
    result["custom_instruction"] = instruction[:500]
    result["lexical_grounding"] = _as_bool(
        source.get("lexical_grounding"), DEFAULT_SETTINGS["lexical_grounding"]
    )
    compute_mode = str(source.get("compute_mode", DEFAULT_SETTINGS["compute_mode"]))
    result["compute_mode"] = compute_mode if compute_mode in COMPUTE_MODES else "auto"
    result["schema"] = SETTINGS_SCHEMA
    return result


def load_settings(path: Optional[str | Path] = None) -> dict[str, Any]:
    settings_path = Path(path) if path else default_settings_path()
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return dict(DEFAULT_SETTINGS)
        return normalize_settings(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: Mapping[str, Any], path: Optional[str | Path] = None) -> Path:
    settings_path = Path(path) if path else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_settings(settings)
    temporary = settings_path.with_suffix(settings_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, settings_path)
    return settings_path


def migrate_legacy_windows_autostart(
    path: Optional[str | Path] = None,
    *,
    registry_module: Any = None,
) -> bool:
    """Remove the old direct-server startup entry and preserve AI autostart.

    Older development installs launched ``YamatanaAIIME.exe --server`` from
    HKCU while the MSI launches the tray from HKLM. When both entries remain,
    the direct server wins the named-pipe race and the tray reports that a
    different ranker owns the pipe. Only the exact legacy command shape is
    migrated so an unrelated value with the same name is left untouched.
    """
    if registry_module is None:
        if os.name != "nt":
            return False
        import winreg as registry_module

    access = registry_module.KEY_QUERY_VALUE | registry_module.KEY_SET_VALUE
    try:
        key = registry_module.OpenKey(
            registry_module.HKEY_CURRENT_USER,
            WINDOWS_RUN_KEY,
            0,
            access,
        )
    except OSError:
        return False

    try:
        try:
            command, _value_type = registry_module.QueryValueEx(
                key, LEGACY_AUTOSTART_VALUE_NAME
            )
        except OSError:
            return False
        normalized = str(command).casefold()
        if not all(
            marker in normalized
            for marker in ("yamatanaaiime.exe", "--server", "ai_ime_ranker")
        ):
            return False
        registry_module.DeleteValue(key, LEGACY_AUTOSTART_VALUE_NAME)
    finally:
        registry_module.CloseKey(key)

    settings = load_settings(path)
    if not settings["ai_autostart"]:
        settings["ai_autostart"] = True
        save_settings(settings, path)
    return True


def domain_instruction(settings: Mapping[str, Any]) -> str:
    normalized = normalize_settings(settings)
    domain = normalized["document_domain"]
    preset = DOMAIN_PRESETS[domain][1]
    custom = normalized["custom_instruction"]
    if domain == "custom":
        return custom or DOMAIN_PRESETS["general"][1]
    if custom:
        return f"{preset} 追加指示: {custom}"
    return preset


def settings_runtime_signature(settings: Mapping[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_settings(settings)
    return (
        normalized["context_enabled"],
        normalized["context_chars"],
        normalized["document_domain"],
        normalized["custom_instruction"],
        normalized["lexical_grounding"],
        normalized["compute_mode"],
    )
