"""Notification-area ON/OFF controller for the distilled Mozc AI ranker."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pystray
from PIL import Image, ImageDraw

from product_settings import (
    COMPUTE_MODES,
    DOMAIN_PRESETS,
    PRODUCT_NAME,
    default_settings_path,
    load_settings,
    migrate_legacy_windows_autostart,
    product_data_dir,
    settings_runtime_signature,
)

PIPE_NAME = r"\\.\pipe\ai_ime_ranker"
PRODUCT_DATA_DIR = product_data_dir()
SETTINGS_FILE = default_settings_path()
STATUS_FILE = PRODUCT_DATA_DIR / "ai_ime_status.json"
LOG_DIR = PRODUCT_DATA_DIR / "logs"
MODEL_LABEL = "Ruri-v3-70M (IME distilled)"


def _packaged_ensemble_arguments(settings: dict[str, Any]) -> list[str]:
    """Return the packaged LoRA3/LoRA6 model arguments when available.

    The development tree still falls back to the normal single-model
    autodetection.  Frozen installations contain both precision variants
    under ``models/onnx``; selecting the pair here keeps the ONNX backend's
    calibrated ensemble active without requiring a user-editable setting.
    """
    roots: list[Path] = []
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(getattr(sys, "_MEIPASS")))
    roots.extend((ROOT, ROOT / "_internal"))
    compute_mode = str(settings.get("compute_mode", "auto"))
    use_gpu = compute_mode in {"auto", "gpu"}
    if use_gpu:
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
            use_gpu = bool(
                available.intersection({"CUDAExecutionProvider", "DmlExecutionProvider"})
            )
        except Exception:
            use_gpu = False
    precision = "fp16" if use_gpu else "int8"
    relative = (
        Path("models") / "onnx" / f"ruri-ime-lora3-{precision}.onnx",
        Path("models") / "onnx" / f"ruri-ime-lora6-{precision}.onnx",
    )
    for root in roots:
        paths = [root / item for item in relative]
        if all(path.exists() for path in paths):
            return [argument for path in paths for argument in ("--ensemble-model", str(path))]
    return []


def make_icon(state: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = {
        "off": ((75, 80, 90), (190, 195, 205)),
        "loading": ((190, 135, 0), (255, 220, 60)),
        "on": ((20, 145, 45), (95, 245, 115)),
        "error": ((170, 25, 35), (255, 115, 125)),
    }
    bg, fg = colors.get(state, colors["off"])
    draw.ellipse([4, 4, 60, 60], fill=bg)
    if state == "on":
        draw.polygon([(32, 9), (20, 36), (30, 36), (32, 55), (45, 27), (35, 27)], fill=fg)
    elif state == "loading":
        draw.arc([12, 12, 52, 52], 25, 305, fill=fg, width=8)
    elif state == "error":
        draw.line([18, 18, 46, 46], fill=fg, width=7)
        draw.line([46, 18, 18, 46], fill=fg, width=7)
    else:
        draw.arc([16, 16, 48, 48], 55, 305, fill=fg, width=6)
        draw.line([32, 11, 32, 33], fill=fg, width=6)
    return img


def _health_request() -> dict[str, Any]:
    return {
        "request_id": f"tray-probe-{time.time_ns()}",
        "inference_trigger": "explicit",
        "preceding_text": "接続確認のため候補を",
        "read": "えらぶ",
        "candidates": [
            {"id": "c0", "text": "選ぶ", "rank": 1},
            {"id": "c1", "text": "撰ぶ", "rank": 2},
        ],
    }


class AIIMETray:
    def __init__(self, allow_legacy_ranker: bool = False) -> None:
        self.state = "off"
        self.server_proc: Optional[subprocess.Popen[bytes]] = None
        self._stdout_handle = None
        self._stderr_handle = None
        self.last_error = ""
        self.allow_legacy_ranker = allow_legacy_ranker
        self.settings = load_settings(SETTINGS_FILE)
        self._settings_signature = settings_runtime_signature(self.settings)
        self._watcher_stop = threading.Event()
        self.icon = pystray.Icon(
            "Yamatana-AI-IME-Mozc",
            make_icon("off"),
            f"{PRODUCT_NAME} [AI OFF]",
            menu=pystray.Menu(
                pystray.MenuItem(
                    self._toggle_text,
                    self.toggle,
                    default=True,
                    checked=self._is_checked,
                    enabled=self._toggle_enabled,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("設定…", self.open_settings),
                pystray.MenuItem("最初の使い方…", self.show_onboarding),
                pystray.MenuItem("状態を確認", self.show_status),
                pystray.MenuItem("プライバシーポリシー", self.show_privacy),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("終了", self.quit_app),
            ),
        )

    def _toggle_text(self, _item: Any) -> str:
        if self.state == "on":
            return "AIを使用する (ON)"
        if self.state == "loading":
            return "70Mモデルを読み込み中…"
        return "AIを使用する (OFF)"

    def _is_checked(self, _item: Any) -> bool:
        return self.state == "on"

    def _toggle_enabled(self, _item: Any) -> bool:
        return self.state != "loading"

    def _set_state(self, state: str, error: str = "") -> None:
        self.state = state
        self.last_error = error
        titles = {
            "off": f"{PRODUCT_NAME} [AI OFF] - 通常変換",
            "loading": f"{PRODUCT_NAME} [読み込み中]",
            "on": f"{PRODUCT_NAME} [AI ON] - 文脈変換",
            "error": f"{PRODUCT_NAME} [エラー] - 状態を確認してください",
        }
        self.icon.icon = make_icon(state)
        self.icon.title = titles[state]
        self.icon.update_menu()

    def _pipe_ready(self, timeout_ms: int = 50) -> bool:
        if sys.platform != "win32":
            return False
        k32 = ctypes.windll.kernel32
        k32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
        k32.WaitNamedPipeW.restype = ctypes.c_int
        return bool(k32.WaitNamedPipeW(PIPE_NAME, max(1, timeout_ms)))

    def _probe(self, timeout_ms: int = 1000) -> bool:
        try:
            from client.windows_pipe import rank_once
            response = rank_once(PIPE_NAME, _health_request(), timeout_ms=timeout_ms)
            return len(response.get("candidates", [])) == 2
        except Exception:
            return False

    def toggle(self, _icon: Any = None, _item: Any = None) -> None:
        if self.state == "loading":
            return
        if self.state == "on":
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        if self._pipe_ready(20):
            if self.allow_legacy_ranker and self._probe(1500):
                self.allow_legacy_ranker = False
                self._set_state("on")
                self.icon.notify(
                    "旧自動起動を移行しました。次回起動からトレイがAIランカーを管理します。",
                    PRODUCT_NAME,
                )
                return
            self._set_state("error", "別のAIランカーが既に同じパイプを使用しています。")
            return
        self._set_state("loading")
        self.settings = load_settings(SETTINGS_FILE)
        self._settings_signature = settings_runtime_signature(self.settings)
        compute_label = COMPUTE_MODES[self.settings["compute_mode"]]
        self.icon.notify(
            f"Ruri 70M蒸留モデルを読み込んでいます。\n演算: {compute_label}",
            PRODUCT_NAME,
        )

        def launch() -> None:
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                if getattr(sys, "frozen", False):
                    cmd = [sys.executable, "--server"]
                else:
                    cmd = [sys.executable, str(ROOT / "ai_ime_tray.py"), "--server"]
                cmd += [
                    "--pipe", PIPE_NAME,
                    "--no-ui",
                    "--settings-file", str(SETTINGS_FILE),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)
                self._stdout_handle = open(LOG_DIR / "ruri_ranker.stdout.log", "ab", buffering=0)
                self._stderr_handle = open(LOG_DIR / "ruri_ranker.stderr.log", "ab", buffering=0)
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                self.server_proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._stdout_handle,
                    stderr=self._stderr_handle,
                    creationflags=flags,
                )
                deadline = time.time() + 120.0
                while time.time() < deadline:
                    if self.server_proc.poll() is not None:
                        raise RuntimeError(f"AIランカーが終了しました (code={self.server_proc.returncode})")
                    if self._pipe_ready(250) and self._probe(1500):
                        self._set_state("on")
                        self.icon.notify("AI変換を有効にしました。", PRODUCT_NAME)
                        self._monitor_server()
                        return
                    time.sleep(0.25)
                raise TimeoutError("70Mモデルの読み込みが120秒以内に完了しませんでした")
            except Exception as exc:
                self._terminate_owned_server()
                self._set_state("error", str(exc))
                self.icon.notify("起動に失敗しました。「状態を確認」を開いてください。", PRODUCT_NAME)

        threading.Thread(target=launch, daemon=True).start()

    def _monitor_server(self) -> None:
        def monitor() -> None:
            proc = self.server_proc
            if proc is None:
                return
            proc.wait()
            if self.state == "on":
                self._set_state("error", f"AIランカーが予期せず終了しました (code={proc.returncode})")
        threading.Thread(target=monitor, daemon=True).start()

    def _terminate_owned_server(self) -> None:
        proc = self.server_proc
        self.server_proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)

    def _stop_server(self) -> None:
        self._terminate_owned_server()
        self._set_state("off")
        self.icon.notify("AI変換を無効にしました。Mozcは通常変換で動作します。", PRODUCT_NAME)

    def _read_status(self) -> dict[str, Any]:
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _launch_self(self, *arguments: str) -> None:
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, *arguments]
        else:
            cmd = [sys.executable, str(ROOT / "ai_ime_tray.py"), *arguments]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )

    def open_settings(self, _icon: Any = None, _item: Any = None) -> None:
        try:
            self._launch_self("--settings")
        except OSError as exc:
            self._set_state("error", f"設定画面を起動できませんでした: {exc}")

    def show_onboarding(self, _icon: Any = None, _item: Any = None) -> None:
        try:
            self._launch_self("--onboarding", "--force-onboarding")
        except OSError as exc:
            self._set_state("error", f"最初の使い方を表示できませんでした: {exc}")

    def _show_first_run_guide(self) -> None:
        try:
            self._launch_self("--onboarding")
        except OSError as exc:
            logging.warning("Could not show first-run guide: %s", exc)

    def show_privacy(self, _icon: Any = None, _item: Any = None) -> None:
        bundle_root = Path(getattr(sys, "_MEIPASS", ROOT))
        candidates = [
            bundle_root / "documents" / "PRIVACY_POLICY_JA.txt",
            ROOT / "documents" / "PRIVACY_POLICY_JA.txt",
            ROOT / "docs" / "PRIVACY_POLICY_JA.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is not None:
            try:
                os.startfile(path)  # type: ignore[attr-defined]
                return
            except OSError:
                pass
        message = (
            "入力内容、前後の文脈、変換候補、利用統計、端末情報を"
            "外部へ送信・収集しません。\n\nAI推論はこのPC内で完結し、"
            "診断ログに候補本文は記録しません。"
        )
        ctypes.windll.user32.MessageBoxW(0, message, f"{PRODUCT_NAME} プライバシー", 0x40)

    def _watch_settings(self) -> None:
        while not self._watcher_stop.wait(1.0):
            latest = load_settings(SETTINGS_FILE)
            signature = settings_runtime_signature(latest)
            if signature == self._settings_signature:
                self.settings = latest
                continue
            self.settings = latest
            self._settings_signature = signature
            if self.state != "on":
                self.icon.update_menu()
                continue
            self._set_state("loading")
            self.icon.notify("設定を反映するためAIモデルを再起動します。", PRODUCT_NAME)
            self._terminate_owned_server()
            self._start_server()

    def show_status(self, _icon: Any = None, _item: Any = None) -> None:
        data = self._read_status()
        settings = load_settings(SETTINGS_FILE)
        domain_label = DOMAIN_PRESETS[settings["document_domain"]][0]
        context_label = (
            f"最大{settings['context_chars']}文字"
            if settings["context_enabled"]
            else "使用しない"
        )
        dictionary_label = "ON" if settings["lexical_grounding"] else "OFF"
        compute_label = COMPUTE_MODES[settings["compute_mode"]]
        if self.state == "on":
            requests = int(data.get("ime_requests", 0))
            reordered = int(data.get("ime_reordered_requests", 0))
            top_changed = int(data.get("ime_top_changed_requests", 0))
            latency = data.get("last_latency_ms")
            last = "まだ実IMEからの要求はありません"
            if requests:
                changed = data.get("last_ime_order_changed")
                promoted = data.get("last_promoted_from_rank")
                context_chars = data.get("last_context_chars")
                candidate_count = data.get("last_candidate_count")
                last_changed = "変更あり" if changed else "変更なし"
                last = (
                    f"実IME要求: {requests}回 / 順位変更: {reordered}回\n"
                    f"第一候補の変更: {top_changed}回\n"
                    f"最終推論: {latency} ms / {last_changed}"
                    f"（元{promoted}位→1位）\n"
                    f"直前文脈: {context_chars}文字\n"
                    f"候補: {candidate_count}件（全候補を1バッチ採点）"
                )
            message = (
                "【AI: ON】\n\n"
                f"モデル: {MODEL_LABEL}\n"
                f"専門分野: {domain_label}\n"
                f"文脈: {context_label} / 辞書認識: {dictionary_label}\n"
                f"演算: {compute_label}\n"
                f"{last}\n\n"
                "入力して変換した後、この画面の実IME要求数が増えれば、\n"
                "インストール済みMozcから70Mモデルまで実際に到達しています。"
            )
        elif self.state == "loading":
            message = "【AI: 読み込み中】\n\nRuri 70M蒸留モデルをGPU/CPUへ読み込んでいます。"
        elif self.state == "error":
            message = f"【AI: エラー】\n\n{self.last_error}\n\nログ: {LOG_DIR}"
        else:
            message = (
                "【AI: OFF】\n\nAIモデルは停止中です。Mozcは通常変換で動作します。\n"
                "モデルのCPU/GPUメモリは使用していません。\n\n"
                f"次回起動時の演算: {compute_label}\n専門分野: {domain_label}"
            )
        ctypes.windll.user32.MessageBoxW(0, message, f"{PRODUCT_NAME} ステータス", 0x40)

    def quit_app(self, _icon: Any = None, _item: Any = None) -> None:
        self._watcher_stop.set()
        self._terminate_owned_server()
        self.icon.stop()

    def run(self, start_on: Optional[bool] = None) -> None:
        threading.Thread(target=self._watch_settings, daemon=True).start()
        if start_on is None:
            start_on = bool(load_settings(SETTINGS_FILE)["ai_autostart"])

        def setup(icon: pystray.Icon) -> None:
            icon.visible = True
            # Let the tray icon settle before the short first-run guide opens,
            # so its instructions point to something that already exists.
            threading.Timer(0.8, self._show_first_run_guide).start()
            if start_on:
                self._start_server()

        self.icon.run(setup=setup)


def run_server_mode(pipe_name: str, settings_file: str | Path = SETTINGS_FILE) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_DIR / "ruri_ranker.service.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )
    try:
        from ranker.ranker import main as ranker_main
        product_settings = load_settings(settings_file)
        ensemble_arguments = _packaged_ensemble_arguments(product_settings)
        return ranker_main([
            "--pipe", pipe_name,
            "--backend", "onnx",
            "--no-ui",
            "--status-file", str(STATUS_FILE),
            "--settings-file", str(settings_file),
            *ensemble_arguments,
        ])
    except Exception as exc:
        logging.exception("Ruri server fatal error: %s", exc)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--start-on", action="store_true")
    parser.add_argument("--settings", action="store_true")
    parser.add_argument("--onboarding", action="store_true")
    parser.add_argument("--force-onboarding", action="store_true")
    parser.add_argument("--pipe", default=PIPE_NAME)
    parser.add_argument("--settings-file", default=str(SETTINGS_FILE))
    parser.add_argument("--from-installer", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    if args.settings:
        from settings_ui import main as settings_main
        return settings_main(SETTINGS_FILE)
    if args.onboarding:
        from onboarding_ui import main as onboarding_main
        return onboarding_main(force=args.force_onboarding)
    if args.server:
        return run_server_mode(args.pipe, args.settings_file)
    if sys.platform == "win32":
        mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "Mozc_AI_IME_LoRA_Tray")
        if ctypes.get_last_error() == 183:
            return 0
        globals()["_TRAY_MUTEX"] = mutex
    migrated_legacy_autostart = migrate_legacy_windows_autostart(SETTINGS_FILE)
    AIIMETray(allow_legacy_ranker=migrated_legacy_autostart).run(
        # The installer passes this flag only on a fresh installation.  Make
        # it explicit so a missing/old settings file cannot leave the first
        # launch in the OFF state.
        start_on=True if (args.start_on or args.from_installer) else None
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
