"""Mini Floating Switch & Control Panel for Mozc AI IME.
Provides a visible, always-on-top taskbar-docked toggle widget + full dashboard.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from product_settings import PRODUCT_NAME

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIPE_NAME = r"\\.\pipe\ai_ime_ranker"

# Legacy entry point kept for existing shortcuts.  Product builds use the
# settings workbench instead of the former floating controller, which could
# terminate unrelated Python processes when switching AI off.
if __name__ == "__main__":
    from settings_ui import main as settings_main
    raise SystemExit(settings_main())


class MozcMiniWidget:
    """Always-on-top compact taskbar widget for 1-click AI IME toggle."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{PRODUCT_NAME} スイッチ")
        self.root.geometry("360x180")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1e1e2e")

        # Center on screen
        self.root.eval('tk::PlaceWindow . center')
        self.root.lift()
        self.root.focus_force()

        self.server_proc: subprocess.Popen | None = None
        self.is_running = False
        self.is_loading = False

        # Header
        lbl_title = tk.Label(
            self.root,
            text=f"⚡ {PRODUCT_NAME} コントローラー",
            font=("Segoe UI", 12, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e",
        )
        lbl_title.pack(pady=(16, 4))

        self.status_lbl = tk.Label(
            self.root,
            text="● 待機中 (OFF / VRAM: 0 MB)",
            font=("Segoe UI", 10),
            fg="#9399b2",
            bg="#1e1e2e",
        )
        self.status_lbl.pack(pady=(0, 12))

        # Main Big Toggle Button
        self.btn = tk.Button(
            self.root,
            text="AI IME を有効化する (ON)",
            font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            activebackground="#94e2d5",
            activeforeground="#11111b",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_toggle,
            padx=16,
            pady=8,
        )
        self.btn.pack(fill=tk.X, padx=32, pady=(0, 12))

        self._check_pipe_status()

    def _start_drag(self, event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event) -> None:
        x = self.root.winfo_x() + (event.x - self._drag_start_x)
        y = self.root.winfo_y() + (event.y - self._drag_start_y)
        self.root.geometry(f"+{x}+{y}")

    def _check_pipe_status(self) -> None:
        try:
            from client.windows_pipe import rank_once
            rank_once(PIPE_NAME, {"request_id": "probe", "preceding_text": "", "read": "", "candidates": []}, timeout_ms=50)
            self._set_ui_on()
        except Exception:
            self._set_ui_off()

    def _set_ui_off(self) -> None:
        self.is_running = False
        self.is_loading = False
        self.status_lbl.config(text="● 待機中 (OFF / VRAM: 0 MB)", fg="#9399b2")
        self.btn.config(text="⚡ AI IME を有効化する (ON)", bg="#a6e3a1", fg="#11111b", state=tk.NORMAL)

    def _set_ui_loading(self) -> None:
        self.is_loading = True
        self.status_lbl.config(text="⏳ GPUにロード中 (約3秒)...", fg="#f9e2af")
        self.btn.config(text="ロード中...", bg="#585b70", fg="#cdd6f4", state=tk.DISABLED)

    def _set_ui_on(self) -> None:
        self.is_running = True
        self.is_loading = False
        self.status_lbl.config(text="● 稼働中 (Ruri-70M / 全候補1バッチ)", fg="#a6e3a1")
        self.btn.config(text="⏹ AI IME を無効化する (OFF)", bg="#f38ba8", fg="#11111b", state=tk.NORMAL)

    def _on_toggle(self) -> None:
        if self.is_running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        self._set_ui_loading()

        def _worker():
            try:
                local_app_data = os.environ.get("LOCALAPPDATA", os.environ.get("TEMP", "."))
                log_dir = Path(local_app_data) / "Mozc" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    str(ROOT / "ranker" / "ranker.py"),
                    "--pipe", PIPE_NAME,
                    "--backend", "ruri",
                    "--no-ui",
                ]
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)

                self.server_proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=flags,
                )
                time.sleep(3.5)
                self.root.after(0, self._set_ui_on)
            except Exception as e:
                self.root.after(0, self._set_ui_off)

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_server(self) -> None:
        if self.server_proc:
            try:
                self.server_proc.terminate()
            except Exception:
                pass
            self.server_proc = None
        subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *ranker*"], capture_output=True)
        self._set_ui_off()

    def run(self) -> None:
        self.root.mainloop()
