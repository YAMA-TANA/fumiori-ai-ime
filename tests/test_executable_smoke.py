"""Smoke test to verify that the packaged PyInstaller binary starts without error."""

import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "YamatanaAIIME" / "YamatanaAIIME.exe"


def test_frozen_executable_smoke() -> None:
    if not EXE.exists():
        pytest.skip("Frozen executable not built yet")
    result = subprocess.run(
        [str(EXE), "--check"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Executable failed with exit code {result.returncode}: {result.stderr}"
