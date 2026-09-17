"""Product settings window for Fumiori AI IME."""

# Hallmark · pre-emit critique: P5 H5 E4 S5 R5 V4
# Hallmark · macrostructure: Workbench · genre: modern-minimal · theme: Cobalt

from __future__ import annotations

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Optional

from product_settings import (
    COMPUTE_MODES,
    CONTEXT_LENGTHS,
    DEFAULT_SETTINGS,
    DOMAIN_PRESETS,
    PRODUCT_NAME,
    default_settings_path,
    load_settings,
    normalize_settings,
    product_data_dir,
    save_settings,
)


if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).resolve().parent


# Named native-UI tokens. The restrained cobalt signal is used only for the
# selected category, focus/primary action, and the AI state explanation.
TOKENS = {
    "paper": "#F7F9FC",
    "surface": "#FFFFFF",
    "surface_alt": "#EFF3F8",
    "ink": "#172033",
    "muted": "#536078",
    "rule": "#D6DEEA",
    "accent": "#245EEA",
    "accent_active": "#1948BD",
    "accent_ink": "#FFFFFF",
    "success": "#117A45",
    "error": "#B42318",
}


class SettingsWindow:
    """A local, keyboard-friendly settings workbench."""

    CATEGORIES = (
        ("ai", "AI"),
        ("context", "文脈"),
        ("domain", "専門分野"),
        ("dictionary", "辞書"),
        ("compute", "演算"),
        ("diagnostics", "診断・プライバシー"),
    )

    def __init__(self, settings_path: Optional[str | Path] = None) -> None:
        self.settings_path = Path(settings_path) if settings_path else default_settings_path()
        self.original = load_settings(self.settings_path)
        self.root = tk.Tk()
        self.root.title(f"{PRODUCT_NAME} — 設定")
        self.root.geometry("820x600")
        self.root.minsize(760, 540)
        self.root.configure(bg=TOKENS["paper"])

        self.ai_autostart = tk.BooleanVar(value=self.original["ai_autostart"])
        self.context_enabled = tk.BooleanVar(value=self.original["context_enabled"])
        self.context_chars = tk.StringVar(value=str(self.original["context_chars"]))
        self.document_domain = tk.StringVar(value=self.original["document_domain"])
        self.lexical_grounding = tk.BooleanVar(value=self.original["lexical_grounding"])
        self.compute_mode = tk.StringVar(value=self.original["compute_mode"])
        self.status_text = tk.StringVar(value="設定はこのPC内だけに保存されます。")
        self.nav_buttons: dict[str, tk.Button] = {}
        self.pages: dict[str, tk.Frame] = {}

        self._configure_styles()
        self._build_layout()
        self._show_page("ai")
        self.root.after(100, lambda: self.nav_buttons["ai"].focus_set())

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        for style_name, widget in (
            ("Yamatana.TCheckbutton", "TCheckbutton"),
            ("Yamatana.TRadiobutton", "TRadiobutton"),
        ):
            style.configure(
                style_name,
                background=TOKENS["surface"],
                foreground=TOKENS["ink"],
                font=("Segoe UI", 10),
                padding=(0, 7),
            )
            style.map(style_name, background=[("active", TOKENS["surface"])])
        style.configure("Yamatana.TCombobox", padding=7)

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=TOKENS["paper"], height=82)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text=PRODUCT_NAME, bg=TOKENS["paper"], fg=TOKENS["ink"],
            font=("Segoe UI Semibold", 20),
        ).pack(anchor=tk.W, padx=28, pady=(17, 0))
        tk.Label(
            header, text="MOZC Ver  ·  ローカルAI変換の設定", bg=TOKENS["paper"],
            fg=TOKENS["muted"], font=("Segoe UI", 9),
        ).pack(anchor=tk.W, padx=30, pady=(1, 0))
        tk.Frame(self.root, bg=TOKENS["rule"], height=1).pack(fill=tk.X)

        content = tk.Frame(self.root, bg=TOKENS["paper"])
        content.pack(fill=tk.BOTH, expand=True)
        nav = tk.Frame(content, bg=TOKENS["surface_alt"], width=190)
        nav.pack(side=tk.LEFT, fill=tk.Y)
        nav.pack_propagate(False)
        for key, label in self.CATEGORIES:
            button = tk.Button(
                nav, text=label, anchor=tk.W, relief=tk.FLAT, bd=0, padx=22, pady=12,
                bg=TOKENS["surface_alt"], fg=TOKENS["ink"],
                activebackground=TOKENS["surface"], activeforeground=TOKENS["accent"],
                font=("Segoe UI", 10), cursor="hand2",
                command=lambda page=key: self._show_page(page),
            )
            button.pack(fill=tk.X, padx=10, pady=(8 if key == "ai" else 0, 0))
            self.nav_buttons[key] = button

        self.page_host = tk.Frame(content, bg=TOKENS["surface"])
        self.page_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for key, _label in self.CATEGORIES:
            page = tk.Frame(self.page_host, bg=TOKENS["surface"])
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[key] = page

        self._build_ai_page()
        self._build_context_page()
        self._build_domain_page()
        self._build_dictionary_page()
        self._build_compute_page()
        self._build_diagnostics_page()

        footer = tk.Frame(self.root, bg=TOKENS["paper"], height=70)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        tk.Label(
            footer, textvariable=self.status_text, bg=TOKENS["paper"],
            fg=TOKENS["muted"], font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=24)
        self._button(footer, "キャンセル", self.root.destroy, False).pack(
            side=tk.RIGHT, padx=(0, 16), pady=14
        )
        self._button(footer, "保存", self._save, True).pack(
            side=tk.RIGHT, padx=(0, 8), pady=14
        )
        self._button(footer, "既定値に戻す", self._restore_defaults, False).pack(
            side=tk.RIGHT, padx=(0, 8), pady=14
        )

    def _page_heading(self, page: tk.Frame, title: str, description: str) -> tk.Frame:
        tk.Label(
            page, text=title, bg=TOKENS["surface"], fg=TOKENS["ink"],
            font=("Segoe UI Semibold", 17),
        ).pack(anchor=tk.W, padx=34, pady=(30, 4))
        tk.Label(
            page, text=description, bg=TOKENS["surface"], fg=TOKENS["muted"],
            font=("Segoe UI", 10), justify=tk.LEFT, wraplength=520,
        ).pack(anchor=tk.W, padx=34, pady=(0, 20))
        body = tk.Frame(page, bg=TOKENS["surface"])
        body.pack(fill=tk.BOTH, expand=True, padx=34)
        return body

    def _card(self, parent: tk.Widget, title: str, detail: str = "") -> tk.Frame:
        card = tk.Frame(
            parent, bg=TOKENS["surface"], highlightbackground=TOKENS["rule"],
            highlightthickness=1, padx=18, pady=14,
        )
        card.pack(fill=tk.X, pady=(0, 14))
        tk.Label(
            card, text=title, bg=TOKENS["surface"], fg=TOKENS["ink"],
            font=("Segoe UI Semibold", 11),
        ).pack(anchor=tk.W)
        if detail:
            tk.Label(
                card, text=detail, bg=TOKENS["surface"], fg=TOKENS["muted"],
                font=("Segoe UI", 9), justify=tk.LEFT, wraplength=500,
            ).pack(anchor=tk.W, pady=(4, 8))
        return card

    def _build_ai_page(self) -> None:
        body = self._page_heading(
            self.pages["ai"], "AI", "トレイは常駐しますが、AIモデルは既定では起動しません。",
        )
        card = self._card(
            body, "AIモデルの常駐",
            "トレイの「AIを使用する」を押した時だけモデルを読み込みます。OFFにするとモデルプロセスを終了し、CPU/GPUメモリを解放します。",
        )
        ttk.Checkbutton(
            card, text="Windowsへのサインイン後、自動的にAIも有効にする",
            variable=self.ai_autostart, style="Yamatana.TCheckbutton",
        ).pack(anchor=tk.W)
        tk.Label(
            card, text="既定値: OFF（トレイのみ起動）", bg=TOKENS["surface"],
            fg=TOKENS["accent"], font=("Segoe UI Semibold", 9),
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_context_page(self) -> None:
        body = self._page_heading(
            self.pages["context"], "文脈", "直前の文章を候補順位の判断に使う範囲を指定します。",
        )
        card = self._card(
            body, "文章の文脈を使う",
            "文脈は推論中だけメモリで処理され、本文としてログや設定ファイルには保存されません。",
        )
        ttk.Checkbutton(
            card, text="直前文脈をAI変換に使用する", variable=self.context_enabled,
            style="Yamatana.TCheckbutton", command=self._sync_context_state,
        ).pack(anchor=tk.W)
        tk.Label(
            card, text="保持する最大文字数", bg=TOKENS["surface"], fg=TOKENS["ink"],
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(12, 4))
        self.context_combo = ttk.Combobox(
            card, textvariable=self.context_chars,
            values=[str(value) for value in CONTEXT_LENGTHS if value > 0],
            state="readonly", width=18, style="Yamatana.TCombobox",
        )
        self.context_combo.pack(anchor=tk.W)
        tk.Label(
            card, text="推奨: 128文字。長くすると判断材料が増えますが、処理時間も増えます。",
            bg=TOKENS["surface"], fg=TOKENS["muted"], font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))
        self._sync_context_state()

    def _build_domain_page(self) -> None:
        body = self._page_heading(
            self.pages["domain"], "専門分野", "文書の種類をAIへのローカル指示として毎回渡します。",
        )
        card = self._card(
            body, "文書の分野",
            "例えば「医学・医療」を選ぶと、医学用語と正式な医療表記を優先するよう候補を評価します。",
        )
        self.domain_by_label = {label: key for key, (label, _prompt) in DOMAIN_PRESETS.items()}
        self.domain_label = tk.StringVar(value=DOMAIN_PRESETS[self.document_domain.get()][0])
        domain_combo = ttk.Combobox(
            card, textvariable=self.domain_label, values=list(self.domain_by_label),
            state="readonly", width=28, style="Yamatana.TCombobox",
        )
        domain_combo.pack(anchor=tk.W)
        domain_combo.bind("<<ComboboxSelected>>", self._domain_changed)
        tk.Label(
            card, text="追加の指示（任意、最大500文字）", bg=TOKENS["surface"],
            fg=TOKENS["ink"], font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(14, 4))
        self.custom_instruction = tk.Text(
            card, height=5, wrap=tk.WORD, undo=True, relief=tk.FLAT, bd=0,
            highlightbackground=TOKENS["rule"], highlightcolor=TOKENS["accent"],
            highlightthickness=1, bg=TOKENS["paper"], fg=TOKENS["ink"],
            insertbackground=TOKENS["ink"], font=("Segoe UI", 10), padx=10, pady=8,
        )
        self.custom_instruction.pack(fill=tk.X)
        self.custom_instruction.insert("1.0", self.original["custom_instruction"])
        tk.Label(
            card, text="例: 循環器内科の診療記録。薬剤名は正式名称を優先する。",
            bg=TOKENS["surface"], fg=TOKENS["muted"], font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(5, 0))

    def _build_dictionary_page(self) -> None:
        body = self._page_heading(
            self.pages["dictionary"], "辞書", "AI評価に加えて、既知の日本語語彙で不自然な表記を抑制します。",
        )
        card = self._card(
            body, "辞書認識",
            "Mozc OSS辞書、文化庁の異字同訓資料、Apache 2.0の同音異義語データから作成したローカル辞書を使います。",
        )
        ttk.Checkbutton(
            card, text="辞書に基づく表記チェックを有効にする",
            variable=self.lexical_grounding, style="Yamatana.TCheckbutton",
        ).pack(anchor=tk.W)
        db_candidates = [
            ROOT / "data" / "massive_homophone_database.json",
            Path(getattr(sys, "_MEIPASS", ROOT)) / "data" / "massive_homophone_database.json",
        ]
        db_path = next((path for path in db_candidates if path.exists()), None)
        state = f"認識済み · {db_path.stat().st_size / (1024 * 1024):.1f} MB" if db_path else "辞書ファイルが見つかりません"
        state_color = TOKENS["success"] if db_path else TOKENS["error"]
        tk.Label(
            card, text=state, bg=TOKENS["surface"], fg=state_color,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor=tk.W, pady=(6, 0))

    def _build_compute_page(self) -> None:
        body = self._page_heading(
            self.pages["compute"], "演算", "AIモデルをどのデバイスで実行するか選択します。",
        )
        card = self._card(
            body, "実行デバイス",
            "自動選択が推奨です。GPUを利用できない場合はCPUへ安全に切り替えます。",
        )
        for value, label in COMPUTE_MODES.items():
            ttk.Radiobutton(
                card, text=label, value=value, variable=self.compute_mode,
                style="Yamatana.TRadiobutton",
            ).pack(anchor=tk.W)
        tk.Label(
            card, text="設定変更時、AIがONならモデルを一度再起動します。IMEはその間も通常変換できます。",
            bg=TOKENS["surface"], fg=TOKENS["muted"], font=("Segoe UI", 9),
            wraplength=500, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))

    def _build_diagnostics_page(self) -> None:
        body = self._page_heading(
            self.pages["diagnostics"], "診断・プライバシー", "すべてのAI処理はこのPCの中で完結します。",
        )
        card = self._card(
            body, "収集しない設計",
            "入力内容、前後の文脈、変換候補、利用統計、端末情報を外部へ送信・収集しません。ネット接続なしでもAI変換できます。診断ログには候補本文を記録しません。",
        )
        self._button(card, "プライバシーポリシーを開く", self._open_privacy, False).pack(
            anchor=tk.W, pady=(4, 0)
        )
        paths = self._card(body, "ローカル保存先")
        tk.Label(
            paths, text=f"設定・ログ: {product_data_dir()}", bg=TOKENS["surface"],
            fg=TOKENS["muted"], font=("Consolas", 9), wraplength=500, justify=tk.LEFT,
        ).pack(anchor=tk.W)
        self._button(paths, "フォルダーを開く", self._open_data_dir, False).pack(
            anchor=tk.W, pady=(10, 0)
        )

    def _button(self, parent: tk.Widget, label: str, command: Any, primary: bool) -> tk.Button:
        background = TOKENS["accent"] if primary else TOKENS["surface"]
        foreground = TOKENS["accent_ink"] if primary else TOKENS["ink"]
        active = TOKENS["accent_active"] if primary else TOKENS["surface_alt"]
        return tk.Button(
            parent, text=label, command=command, relief=tk.FLAT, bd=0,
            highlightbackground=TOKENS["rule"], highlightthickness=1,
            bg=background, fg=foreground, activebackground=active,
            activeforeground=foreground, font=("Segoe UI Semibold", 9),
            padx=16, pady=9, cursor="hand2", takefocus=True,
        )

    def _show_page(self, key: str) -> None:
        self.pages[key].tkraise()
        for page, button in self.nav_buttons.items():
            selected = page == key
            button.configure(
                bg=TOKENS["surface"] if selected else TOKENS["surface_alt"],
                fg=TOKENS["accent"] if selected else TOKENS["ink"],
                font=("Segoe UI Semibold" if selected else "Segoe UI", 10),
            )

    def _sync_context_state(self) -> None:
        self.context_combo.configure(state="readonly" if self.context_enabled.get() else "disabled")

    def _domain_changed(self, _event: Any = None) -> None:
        self.document_domain.set(self.domain_by_label.get(self.domain_label.get(), "general"))

    def _collect(self) -> dict[str, Any]:
        self._domain_changed()
        data = {
            "ai_autostart": self.ai_autostart.get(),
            "context_enabled": self.context_enabled.get(),
            "context_chars": int(self.context_chars.get()) if self.context_enabled.get() else 0,
            "document_domain": self.document_domain.get(),
            "custom_instruction": self.custom_instruction.get("1.0", tk.END).strip(),
            "lexical_grounding": self.lexical_grounding.get(),
            "compute_mode": self.compute_mode.get(),
        }
        return normalize_settings(data)

    def _save(self) -> None:
        try:
            settings = self._collect()
            if settings["document_domain"] == "custom" and not settings["custom_instruction"]:
                self._show_page("domain")
                self.custom_instruction.focus_set()
                messagebox.showerror(
                    PRODUCT_NAME, "カスタム分野を使う場合は、追加の指示を入力してください。",
                    parent=self.root,
                )
                return
            save_settings(settings, self.settings_path)
            self.status_text.set("保存しました。AIがONの場合は自動的に再起動して反映します。")
            self.root.after(650, self.root.destroy)
        except (OSError, ValueError) as exc:
            self.status_text.set("保存できませんでした。")
            messagebox.showerror(PRODUCT_NAME, f"設定を保存できませんでした。\n\n{exc}", parent=self.root)

    def _restore_defaults(self) -> None:
        defaults = dict(DEFAULT_SETTINGS)
        self.ai_autostart.set(defaults["ai_autostart"])
        self.context_enabled.set(defaults["context_enabled"])
        self.context_chars.set(str(defaults["context_chars"]))
        self.document_domain.set(defaults["document_domain"])
        self.domain_label.set(DOMAIN_PRESETS[defaults["document_domain"]][0])
        self.lexical_grounding.set(defaults["lexical_grounding"])
        self.compute_mode.set(defaults["compute_mode"])
        self.custom_instruction.delete("1.0", tk.END)
        self._sync_context_state()
        self.status_text.set("既定値に戻しました。保存すると反映されます。")

    def _open_privacy(self) -> None:
        bundle_root = Path(getattr(sys, "_MEIPASS", ROOT))
        candidates = [
            bundle_root / "documents" / "PRIVACY_POLICY_JA.txt",
            ROOT / "documents" / "PRIVACY_POLICY_JA.txt",
            ROOT / "docs" / "PRIVACY_POLICY_JA.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            messagebox.showinfo(
                PRODUCT_NAME,
                "本製品は入力内容、文脈、候補、利用統計、端末情報を外部へ送信・収集しません。",
                parent=self.root,
            )
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def _open_data_dir(self) -> None:
        directory = product_data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        os.startfile(directory)  # type: ignore[attr-defined]

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(settings_path: Optional[str | Path] = None) -> int:
    return SettingsWindow(settings_path=settings_path).run()


if __name__ == "__main__":
    raise SystemExit(main())
