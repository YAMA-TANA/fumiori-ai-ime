from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from product_settings import (
    DEFAULT_SETTINGS,
    LEGACY_AUTOSTART_VALUE_NAME,
    domain_instruction,
    load_settings,
    migrate_legacy_windows_autostart,
    normalize_settings,
    save_settings,
    settings_runtime_signature,
)


class FakeRegistry:
    HKEY_CURRENT_USER = object()
    KEY_QUERY_VALUE = 1
    KEY_SET_VALUE = 2

    def __init__(self, command: str) -> None:
        self.command = command
        self.deleted: list[str] = []
        self.closed = False

    def OpenKey(self, *_args):
        return object()

    def QueryValueEx(self, _key, _name):
        return self.command, 1

    def DeleteValue(self, _key, name):
        self.deleted.append(name)

    def CloseKey(self, _key):
        self.closed = True


class ProductSettingsTests(unittest.TestCase):
    def test_missing_file_uses_ai_on_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings = load_settings(Path(temp) / "missing.json")
        self.assertTrue(settings["ai_autostart"])
        self.assertTrue(settings["context_enabled"])
        self.assertEqual(settings["context_chars"], 128)
        self.assertEqual(settings["compute_mode"], "auto")

    def test_invalid_values_are_normalized(self) -> None:
        settings = normalize_settings(
            {
                "ai_autostart": "yes",
                "context_chars": 140,
                "document_domain": "unknown",
                "compute_mode": "quantum",
                "custom_instruction": "x" * 800,
            }
        )
        self.assertTrue(settings["ai_autostart"])
        self.assertEqual(settings["context_chars"], 128)
        self.assertEqual(settings["document_domain"], "general")
        self.assertEqual(settings["compute_mode"], "auto")
        self.assertEqual(len(settings["custom_instruction"]), 500)

    def test_settings_round_trip_is_local_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            saved = dict(DEFAULT_SETTINGS)
            saved.update(
                document_domain="medical",
                custom_instruction="循環器内科。薬剤名は正式名称を優先する。",
                compute_mode="cpu",
            )
            save_settings(saved, path)
            loaded = load_settings(path)
        self.assertEqual(loaded["document_domain"], "medical")
        self.assertIn("循環器内科", loaded["custom_instruction"])
        self.assertEqual(loaded["compute_mode"], "cpu")

    def test_custom_instruction_is_added_to_every_domain(self) -> None:
        instruction = domain_instruction(
            {
                **DEFAULT_SETTINGS,
                "document_domain": "medical",
                "custom_instruction": "循環器内科の記録。",
            }
        )
        self.assertIn("医学・医療文書", instruction)
        self.assertIn("循環器内科の記録", instruction)

    def test_ai_autostart_does_not_require_model_restart(self) -> None:
        first = dict(DEFAULT_SETTINGS)
        second = {**first, "ai_autostart": True}
        self.assertEqual(settings_runtime_signature(first), settings_runtime_signature(second))

    def test_legacy_direct_server_autostart_is_migrated(self) -> None:
        registry = FakeRegistry(
            r'"C:\Program Files (x86)\Yamatana AI IME\ai_runtime\YamatanaAIIME.exe" '
            r'--server --pipe "\\.\pipe\ai_ime_ranker" --no-ui'
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            migrated = migrate_legacy_windows_autostart(
                path, registry_module=registry
            )
            settings = load_settings(path)
        self.assertTrue(migrated)
        self.assertEqual(registry.deleted, [LEGACY_AUTOSTART_VALUE_NAME])
        self.assertTrue(registry.closed)
        self.assertTrue(settings["ai_autostart"])

    def test_unrelated_autostart_value_is_not_removed(self) -> None:
        registry = FakeRegistry(r'"C:\Tools\other.exe" --server')
        with tempfile.TemporaryDirectory() as temp:
            migrated = migrate_legacy_windows_autostart(
                Path(temp) / "settings.json", registry_module=registry
            )
        self.assertFalse(migrated)
        self.assertEqual(registry.deleted, [])
        self.assertTrue(registry.closed)


if __name__ == "__main__":
    unittest.main()
