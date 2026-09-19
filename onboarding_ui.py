"""First-run guide for Fumiori AI IME.

Hallmark · pre-emit critique: P5 H5 E5 S5 R5 V5
Hallmark · macrostructure: Narrative Workflow · genre: playful · theme: Hum
Hallmark · audience: first-time Windows IME users · use: discover tray AI toggle
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from typing import Optional

from product_settings import PRODUCT_NAME, PRODUCT_VERSION, product_data_dir


# Tk requires sRGB colours.  Each runtime value is derived from the named Hum
# OKLCH token noted alongside it; keeping the source colour here prevents
# untracked, one-off UI colours from creeping into the native surface.
TOKENS = {
    "paper": "#FAF7EB",       # oklch(97% 0.012 95)
    "paper_2": "#F2EDD9",     # oklch(94% 0.016 95)
    "surface": "#FFFDF5",     # raised cream surface
    "ink": "#24262D",         # oklch(20% 0.012 250)
    "muted": "#5B5E66",       # cool-tinted neutral
    "rule": "#D8D2BE",        # warm hairline
    "pear": "#E9CF45",        # oklch(86% 0.18 95)
    "pear_active": "#D9BD32", # deeper primary press state
    "cyan": "#54A8CB",        # oklch(66% 0.18 235)
    "coral": "#E06E69",       # oklch(68% 0.24 18)
    "mint": "#6FC99C",        # oklch(80% 0.16 150)
    "focus": "#255F8E",       # ≥3:1 focus indicator
}


def onboarding_state_path() -> Path:
    return product_data_dir() / "onboarding.json"


def should_show_onboarding(path: Optional[str | Path] = None) -> bool:
    state_path = Path(path) if path is not None else onboarding_state_path()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    return state.get("completed_for_version") != PRODUCT_VERSION


def mark_onboarding_seen(path: Optional[str | Path] = None) -> Path:
    state_path = Path(path) if path is not None else onboarding_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_for_version": PRODUCT_VERSION,
        "guide": "tray-ai-toggle-v1",
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix="onboarding.", suffix=".tmp", dir=str(state_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, state_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return state_path


def _launch_settings() -> None:
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--settings"]
        working_directory = Path(sys.executable).parent
    else:
        root = Path(__file__).resolve().parent
        command = [sys.executable, str(root / "ai_ime_tray.py"), "--settings"]
        working_directory = root
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        command,
        cwd=str(working_directory),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


class OnboardingWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{PRODUCT_NAME} · 最初の使い方")
        self.root.geometry("680x620")
        self.root.minsize(620, 570)
        self.root.configure(bg=TOKENS["paper"])
        self.root.protocol("WM_DELETE_WINDOW", self.finish)
        self.root.bind("<Escape>", lambda _event: self.finish())
        self._centre()
        self._build()

    def _centre(self) -> None:
        self.root.update_idletasks()
        width, height = 680, 620
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=TOKENS["paper"], padx=42, pady=34)
        shell.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            shell,
            text="まず、ここだけ",
            bg=TOKENS["paper"],
            fg=TOKENS["ink"],
            font=("Yu Gothic UI Semibold", 23),
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            shell,
            text="通常の日本語入力はそのまま使えます。AIは必要な時だけ、\nトレイからONにします。",
            bg=TOKENS["paper"],
            fg=TOKENS["muted"],
            font=("Yu Gothic UI", 10),
            justify=tk.LEFT,
            anchor="w",
            pady=8,
        ).pack(fill=tk.X)

        steps = tk.Frame(shell, bg=TOKENS["paper"], pady=14)
        steps.pack(fill=tk.BOTH, expand=True)
        items = [
            (
                "01",
                "入力方法を選ぶ",
                f"タスクバー右下の「あ / A」、または Win + Space から\n{PRODUCT_NAME} を選びます。",
                TOKENS["pear"],
            ),
            (
                "02",
                "トレイアイコンを見つける",
                "右下の通知領域に丸い電源マークがあります。\n見えない場合は「^」を開きます。",
                TOKENS["cyan"],
            ),
            (
                "03",
                "必要な時だけAIをON",
                "アイコンを右クリックし「AIを使用する (OFF)」を選びます。\nチェックが付き、アイコンが緑になればONです。",
                TOKENS["coral"],
            ),
        ]
        for number, title, detail, accent in items:
            self._step(steps, number, title, detail, accent)

        note = tk.Frame(
            shell,
            bg=TOKENS["paper_2"],
            highlightbackground=TOKENS["rule"],
            highlightthickness=1,
            padx=16,
            pady=12,
        )
        note.pack(fill=tk.X, pady=(4, 18))
        tk.Label(
            note,
            text="AIの初期状態はOFFです",
            bg=TOKENS["paper_2"],
            fg=TOKENS["ink"],
            font=("Yu Gothic UI Semibold", 10),
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            note,
            text="OFFでもMozcの通常変換は使えます。入力内容は外部へ送信しません。",
            bg=TOKENS["paper_2"],
            fg=TOKENS["muted"],
            font=("Yu Gothic UI", 9),
            anchor="w",
        ).pack(fill=tk.X, pady=(3, 0))

        actions = tk.Frame(shell, bg=TOKENS["paper"])
        actions.pack(fill=tk.X)
        self._button(actions, "設定を開く", self.open_settings, primary=False).pack(
            side=tk.LEFT
        )
        self._button(actions, "使い始める", self.finish, primary=True).pack(
            side=tk.RIGHT
        )
        tk.Label(
            shell,
            text="あとでトレイの「最初の使い方…」から再表示できます。",
            bg=TOKENS["paper"],
            fg=TOKENS["muted"],
            font=("Yu Gothic UI", 8),
            anchor="e",
        ).pack(fill=tk.X, pady=(12, 0))

    def _step(
        self,
        parent: tk.Widget,
        number: str,
        title: str,
        detail: str,
        accent: str,
    ) -> None:
        row = tk.Frame(parent, bg=TOKENS["surface"], padx=14, pady=12)
        row.pack(fill=tk.X, pady=(0, 10))
        badge = tk.Label(
            row,
            text=number,
            bg=accent,
            fg=TOKENS["ink"],
            font=("Segoe UI Semibold", 10),
            width=4,
            height=2,
        )
        badge.pack(side=tk.LEFT, padx=(0, 14))
        copy = tk.Frame(row, bg=TOKENS["surface"])
        copy.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            copy,
            text=title,
            bg=TOKENS["surface"],
            fg=TOKENS["ink"],
            font=("Yu Gothic UI Semibold", 11),
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            copy,
            text=detail,
            bg=TOKENS["surface"],
            fg=TOKENS["muted"],
            font=("Yu Gothic UI", 9),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, pady=(3, 0))

    def _button(
        self,
        parent: tk.Widget,
        label: str,
        command: object,
        *,
        primary: bool,
    ) -> tk.Button:
        background = TOKENS["pear"] if primary else TOKENS["surface"]
        active = TOKENS["pear_active"] if primary else TOKENS["paper_2"]
        return tk.Button(
            parent,
            text=label,
            command=command,
            bg=background,
            fg=TOKENS["ink"],
            activebackground=active,
            activeforeground=TOKENS["ink"],
            font=("Yu Gothic UI Semibold", 10),
            relief=tk.FLAT,
            borderwidth=0,
            highlightbackground=TOKENS["focus"],
            highlightthickness=1,
            padx=20,
            pady=10,
            cursor="hand2",
        )

    def open_settings(self) -> None:
        mark_onboarding_seen()
        _launch_settings()
        self.root.destroy()

    def finish(self) -> None:
        mark_onboarding_seen()
        self.root.destroy()

    def run(self) -> int:
        self.root.after(100, self.root.focus_force)
        self.root.mainloop()
        return 0


def main(*, force: bool = False) -> int:
    if not force and not should_show_onboarding():
        return 0
    if sys.platform == "win32":
        mutex = ctypes.windll.kernel32.CreateMutexW(
            None, True, "Yamatana_AI_IME_Onboarding_Guide"
        )
        if ctypes.get_last_error() == 183:
            return 0
        globals()["_ONBOARDING_MUTEX"] = mutex
    return OnboardingWindow().run()


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
