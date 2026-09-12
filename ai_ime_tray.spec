# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs
from scripts.release_version import (
    DEFAULT_PRODUCT_VERSION,
    DEFAULT_RELEASE_LABEL,
    write_pyinstaller_version_info,
)

ROOT = Path(os.path.abspath(".")).resolve()
VERSION_INFO = write_pyinstaller_version_info(
    ROOT / "build" / "pyinstaller-version-info.txt",
    os.environ.get("YAMATANA_PRODUCT_VERSION", DEFAULT_PRODUCT_VERSION),
    os.environ.get("YAMATANA_RELEASE_LABEL", DEFAULT_RELEASE_LABEL),
)
INSTALLED_RUNTIME = Path(os.environ.get(
    "YAMATANA_INSTALLED_RUNTIME",
    r"C:\Program Files (x86)\Yamatana AI IME\ai_runtime\_internal",
))


def asset(local_path, installed_path):
    local = ROOT / local_path
    if local.exists():
        return str(local)
    installed = INSTALLED_RUNTIME / installed_path
    if installed.exists():
        return str(installed)
    return str(local)


def staged_asset(local_path, installed_path, output_name):
    """Copy a model to a stable package name before PyInstaller collects it."""
    source = Path(asset(local_path, installed_path))
    staged = ROOT / "build" / "pyinstaller-models" / output_name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    return str(staged)

block_cipher = None

cuda_binaries = []
for cuda_package in (
    'nvidia.cuda_runtime',
    'nvidia.cuda_nvrtc',
    'nvidia.cublas',
    'nvidia.cudnn',
):
    cuda_binaries.extend(collect_dynamic_libs(cuda_package))

all_datas = [
    (staged_asset('build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx', 'models/onnx/ruri-ime-lora3-fp16.onnx', 'ruri-ime-lora3-fp16.onnx'), 'models/onnx'),
    (staged_asset('build/onnx-model-70m-lora3-20260909/ruri-ime-int8.onnx', 'models/onnx/ruri-ime-lora3-int8.onnx', 'ruri-ime-lora3-int8.onnx'), 'models/onnx'),
    (staged_asset('build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx', 'models/onnx/ruri-ime-lora6-fp16.onnx', 'ruri-ime-lora6-fp16.onnx'), 'models/onnx'),
    (staged_asset('build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-int8.onnx', 'models/onnx/ruri-ime-lora6-int8.onnx', 'ruri-ime-lora6-int8.onnx'), 'models/onnx'),
    (asset('build/onnx-model-70m/tokenizer.json', 'models/onnx/tokenizer.json'), 'models/onnx'),
    (asset('data/massive_homophone_database.json', 'data/massive_homophone_database.json'), 'data'),
    ('PRIVACY.md', 'documents'),
    ('LICENSE', 'documents'),
    ('NOTICE', 'documents'),
    ('THIRD_PARTY_LICENSES.md', 'documents'),
    ('docs/SYSTEM_REQUIREMENTS_JA.md', 'documents'),
    ('docs/README_DISTRIBUTION_JA.md', 'documents'),
]

all_hidden = [
    'pystray',
    'pystray._win32',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'numpy',
    'onnxruntime',
    'onnxruntime.capi._pybind_state',
    'tokenizers',
    'product_settings',
    'settings_ui',
    'onboarding_ui',
    'ranker.ranker',
    'ranker.onnx_ranker',
    'ranker.lexicon',
    'ranker.scoring',
    'ranker.protocol',
    'ranker.loading_ui',
    'client.windows_pipe',
    'nvidia.cuda_runtime',
    'nvidia.cuda_nvrtc',
    'nvidia.cublas',
    'nvidia.cudnn',
]

a = Analysis(
    ['ai_ime_tray.py'],
    pathex=[str(ROOT)],
    binaries=cuda_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'transformers', 'matplotlib', 'scipy', 'pandas', 'nltk', 'pygame',
        'diffusers', 'yt_dlp', 'librosa', 'datasets', 'pyarrow', 'sqlalchemy',
        'uvicorn', 'fastapi', 'starlette', 'rich', 'optuna', 'lightning',
        'aiohttp', 'aiofiles', 'gradio', 'gradio_client', 'openai', 'typer',
        'soundfile', 'torchaudio', 'torchvision', 'trio', 'anyio', 'httpx', 'httpcore',
        'peft', 'tensorboard', 'huggingface_hub', 'requests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YamatanaAIIME',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    version=str(VERSION_INFO),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='YamatanaAIIME',
)
