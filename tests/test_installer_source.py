from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_ACTION = (
    ROOT / "mozc" / "overlay" / "src" / "win32" / "custom_action" / "custom_action.cc"
)
INSTALLER_WXS = ROOT / "mozc" / "overlay" / "src" / "win32" / "installer" / "installer_oss_64bit.wxs"
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


def test_msi_launches_fresh_ai_tray_after_install() -> None:
    installer = INSTALLER_WXS.read_text(encoding="utf-8")
    builder = MSI_BUILDER.read_text(encoding="utf-8")

    assert 'Id="LaunchYamatanaTray"' in installer
    assert 'Action="LaunchYamatanaTray"' in installer
    assert 'Condition="(ACTION=&quot;INSTALL&quot;)"' in installer
    assert "installer_text = installer_text.replace(" not in builder
    assert "installer_oss_64bit.wxs" in builder
