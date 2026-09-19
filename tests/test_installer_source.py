from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ACTION = (
    ROOT / "mozc" / "overlay" / "src" / "win32" / "custom_action" / "custom_action.cc"
)
MSI_BUILDER = ROOT / "scripts" / "build_ai_msi.py"


def test_tip_registration_uses_msi_install_directory() -> None:
    source = CUSTOM_ACTION.read_text(encoding="utf-8")

    assert 'GetProperty(msi_handle, L"CustomActionData")' in source
    assert 'GetProperty(msi_handle, L"MozcDir")' in source
    assert 'SetProperty(msi_handle, L"RegisterTIP64", install_dir)' in source
    assert (
        'SetProperty(msi_handle, L"EnsureAllApplicationPackagesPermisssions",'
        in source
    )
    assert "GetMozcComponentPath(msi_handle, mozc::kMozcTIP32)" in source


def test_msi_completion_dialog_explains_restart_requirement() -> None:
    source = MSI_BUILDER.read_text(encoding="utf-8")

    assert 'WIXUI_EXITDIALOGOPTIONALTEXT' in source
    assert "Windows の再起動が必要です" in source
