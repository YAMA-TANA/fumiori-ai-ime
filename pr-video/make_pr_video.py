from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 720, 1280
FPS = 30
DURATION = 25.0

BG = (8, 11, 14)
PANEL = (18, 23, 29)
PANEL_2 = (23, 30, 37)
WHITE = (245, 248, 250)
MUTED = (151, 164, 176)
GREEN = (0, 208, 82)
GREEN_DARK = (0, 105, 51)
CYAN = (57, 217, 255)
RED = (255, 91, 96)
YELLOW = (255, 207, 74)

FONT_B = r"C:\Windows\Fonts\YuGothB.ttc"
FONT_M = r"C:\Windows\Fonts\YuGothM.ttc"
FONT_LATIN_B = r"C:\Windows\Fonts\seguisb.ttf"
FONT_LATIN = r"C:\Windows\Fonts\segoeui.ttf"


def font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_LATIN_B if latin and bold else FONT_LATIN if latin else FONT_B if bold else FONT_M
    return ImageFont.truetype(path, size=size)


F = {
    "tiny": font(18),
    "small": font(22),
    "small_b": font(22, True),
    "body": font(28),
    "body_b": font(28, True),
    "mid": font(38, True),
    "large": font(54, True),
    "hero": font(76, True),
    "huge": font(100, True),
    "latin_small": font(18, True, True),
    "latin_mid": font(34, True, True),
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def ease(value: float) -> float:
    x = clamp(value)
    return 1.0 - (1.0 - x) ** 3


def smooth(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def span(t: float, start: float, duration: float) -> float:
    return clamp((t - start) / duration)


def pulse(t: float, speed: float = 1.0) -> float:
    return 0.5 + 0.5 * math.sin(t * math.tau * speed)


def alpha_color(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (*color, int(clamp(alpha / 255) * 255))


def rgba_layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def base_frame(t: float) -> Image.Image:
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    vignette = np.sqrt((x - 0.5) ** 2 + ((y - 0.48) * 0.7) ** 2)
    glow = np.exp(-(((x - 0.15 - 0.03 * math.sin(t * 0.35)) / 0.42) ** 2 + ((y - 0.16) / 0.34) ** 2))
    glow2 = np.exp(-(((x - 0.85) / 0.5) ** 2 + ((y - 0.82) / 0.42) ** 2))
    arr = np.zeros((H, W, 3), dtype=np.float32)
    arr[:] = BG
    arr[:, :, 1] += glow * 10 + glow2 * 4
    arr[:, :, 2] += glow * 8 + glow2 * 9
    arr -= np.clip(vignette - 0.35, 0, 1)[:, :, None] * 12
    noise = (np.sin(np.arange(H)[:, None] * 0.47 + t) * 0.45).astype(np.float32)
    arr += noise[:, :, None]
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)), "RGB").convert("RGBA")


def draw_glow_circle(image: Image.Image, xy: tuple[int, int], radius: int, color: tuple[int, int, int], opacity: int = 100) -> None:
    layer = rgba_layer()
    d = ImageDraw.Draw(layer)
    cx, cy = xy
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*color, opacity))
    layer = layer.filter(ImageFilter.GaussianBlur(radius // 2))
    image.alpha_composite(layer)


def round_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill, outline=None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=fnt)


def centered(draw: ImageDraw.ImageDraw, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill=WHITE, x_offset: int = 0) -> None:
    width = text_width(draw, text, fnt)
    draw.text(((W - width) / 2 + x_offset, y), text, font=fnt, fill=fill)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color=GREEN, width: int | None = None) -> int:
    fnt = F["latin_small"]
    tw = int(text_width(draw, text, fnt))
    box_w = width or tw + 28
    round_rect(draw, (x, y, x + box_w, y + 38), 19, (*color, 28), (*color, 100), 1)
    draw.text((x + (box_w - tw) / 2, y + 7), text, font=fnt, fill=color)
    return box_w


def draw_header(draw: ImageDraw.ImageDraw, active: bool = True) -> None:
    draw.ellipse((36, 41, 62, 67), fill=GREEN if active else MUTED)
    draw.polygon([(48, 44), (42, 56), (49, 56), (45, 65), (57, 52), (50, 52)], fill=(7, 45, 23))
    draw.text((76, 42), "Yamatana AI IME", font=F["small_b"], fill=WHITE)
    status = "AI ON" if active else "AI OFF"
    sw = int(text_width(draw, status, F["latin_small"])) + 28
    round_rect(draw, (W - 36 - sw, 38, W - 36, 73), 18, (*GREEN, 28) if active else (*MUTED, 20))
    draw.text((W - 36 - sw + 14, 46), status, font=F["latin_small"], fill=GREEN if active else MUTED)


def draw_keyboard_hint(draw: ImageDraw.ImageDraw, x: int, y: int, text: str = "SPACE") -> None:
    round_rect(draw, (x, y, x + 108, y + 44), 10, (32, 39, 47), (74, 86, 98), 2)
    tw = text_width(draw, text, F["latin_small"])
    draw.text((x + (108 - tw) / 2, y + 10), text, font=F["latin_small"], fill=WHITE)


def draw_candidate_list(
    image: Image.Image,
    x: int,
    y: int,
    items: list[str],
    selected: int = 0,
    alpha: float = 1.0,
    badges: dict[int, str] | None = None,
) -> None:
    layer = rgba_layer()
    d = ImageDraw.Draw(layer)
    row_h = 66
    width = 310
    round_rect(d, (x, y, x + width, y + row_h * len(items) + 20), 22, (25, 31, 39, int(245 * alpha)), (92, 105, 118, int(90 * alpha)), 1)
    for i, item in enumerate(items):
        top = y + 10 + i * row_h
        if i == selected:
            round_rect(d, (x + 8, top, x + width - 8, top + row_h - 4), 14, (*GREEN, int(48 * alpha)), (*GREEN, int(150 * alpha)), 2)
        d.text((x + 24, top + 15), str(i + 1), font=F["small"], fill=(*MUTED, int(255 * alpha)))
        d.text((x + 66, top + 11), item, font=F["body_b"], fill=(*WHITE, int(255 * alpha)))
        if badges and i in badges:
            badge = badges[i]
            bw = int(text_width(d, badge, F["tiny"])) + 18
            round_rect(d, (x + width - bw - 18, top + 14, x + width - 18, top + 47), 16, (*GREEN, int(36 * alpha)))
            d.text((x + width - bw - 9, top + 19), badge, font=F["tiny"], fill=(*GREEN, int(255 * alpha)))
    image.alpha_composite(layer)


def scene_hook(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    p = ease(span(t, 0.10, 0.55))
    y = int(215 + (1 - p) * 55)
    centered(draw, y, "日本語変換は、", F["large"], MUTED)
    centered(draw, y + 80, "まだ文脈を", F["hero"], WHITE)
    centered(draw, y + 174, "読めていない。", F["hero"], WHITE)
    underline_w = int(390 * ease(span(t, 0.72, 0.55)))
    draw.rounded_rectangle((165, y + 270, 165 + underline_w, y + 280), radius=5, fill=GREEN)

    p2 = ease(span(t, 1.22, 0.45))
    if p2 > 0:
        box_y = int(700 + (1 - p2) * 42)
        round_rect(draw, (62, box_y, W - 62, box_y + 185), 26, PANEL, (55, 66, 77), 1)
        draw.text((92, box_y + 30), "しこう", font=F["huge"], fill=WHITE)
        x = 92
        for i, word in enumerate(["思考", "施行", "試行", "志向", "指向"]):
            bw = int(text_width(draw, word, F["small_b"])) + 30
            round_rect(draw, (x, box_y + 124, x + bw, box_y + 168), 20, PANEL_2, (67, 80, 92), 1)
            draw.text((x + 15, box_y + 133), word, font=F["small_b"], fill=MUTED)
            x += bw + 9
    q = ease(span(t, 2.03, 0.42))
    if q:
        centered(draw, 1030, "でも、答えは文章の中にある。", F["mid"], (*WHITE, int(255 * q)))


def scene_legal(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    label(draw, 36, 112, "REAL MODEL DEMO", GREEN)
    draw.text((36, 180), "5位の正解を、", font=F["large"], fill=WHITE)
    draw.text((36, 248), "文脈だけで1位へ。", font=F["large"], fill=WHITE)

    card_y = 365
    round_rect(draw, (36, card_y, W - 36, card_y + 184), 26, PANEL, (54, 66, 77), 1)
    draw.text((66, card_y + 26), "改正された法律は来月から", font=F["body"], fill=MUTED)
    reading = "しこう"
    chars = max(0, min(len(reading), int(span(t, 0.55, 0.72) * (len(reading) + 1))))
    typed = reading[:chars]
    draw.text((66, card_y + 79), typed, font=F["mid"], fill=WHITE)
    if t < 2.55 and pulse(t, 2.2) > 0.35:
        cx = int(66 + text_width(draw, typed, F["mid"]))
        draw.rectangle((cx + 3, card_y + 84, cx + 6, card_y + 128), fill=GREEN)
    draw.text((66 + int(text_width(draw, reading, F["mid"])) + 14, card_y + 87), "される。", font=F["body"], fill=MUTED)

    if t > 1.18:
        draw_keyboard_hint(draw, W - 183, card_y + 126)
    if 1.55 < t < 3.2:
        a = ease(span(t, 1.55, 0.26))
        draw_candidate_list(image, 205, 575, ["思考", "試行", "志向", "指向", "施行"], selected=0, alpha=a)
        if t > 2.05:
            scan_y = int(595 + ((t - 2.05) / 0.75 % 1) * 300)
            glow = rgba_layer()
            gd = ImageDraw.Draw(glow)
            gd.rectangle((215, scan_y, 505, scan_y + 4), fill=(*GREEN, 180))
            glow = glow.filter(ImageFilter.GaussianBlur(7))
            image.alpha_composite(glow)
            centered(draw, 955, "AIが前後の文脈を評価中…", F["small_b"], GREEN)

    if t >= 3.2:
        a = ease(span(t, 3.2, 0.26))
        draw_candidate_list(image, 205, 575, ["施行", "試行", "指向", "志向", "思考"], selected=0, alpha=a, badges={0: "AI 1位"})
        label(draw, 240, 948, "5th  →  1st", GREEN, 240)
        draw.text((121, 1015), "施行", font=F["hero"], fill=GREEN)
        draw.text((286, 1035), "法律を実施する", font=F["body"], fill=WHITE)
        draw.text((109, 1130), "実測 111 ms", font=F["body_b"], fill=WHITE)
        draw.text((330, 1137), "※CPU・候補5件・ウォームアップ後", font=F["tiny"], fill=MUTED)


def context_card(draw: ImageDraw.ImageDraw, y: int, category: str, prefix: str, answer: str, suffix: str, color, progress: float) -> None:
    slide = int((1 - ease(progress)) * 80)
    x = 36 + slide
    alpha = int(255 * ease(progress))
    round_rect(draw, (x, y, W - 36 + slide, y + 166), 24, (18, 24, 30, alpha), (62, 73, 84, min(alpha, 130)), 1)
    pill_w = int(text_width(draw, category, F["small_b"])) + 30
    round_rect(draw, (x + 22, y + 20, x + 22 + pill_w, y + 59), 19, (*color, min(alpha, 42)), (*color, min(alpha, 130)), 1)
    draw.text((x + 37, y + 27), category, font=F["small_b"], fill=(*color, alpha))
    line_y = y + 84
    draw.text((x + 26, line_y), prefix, font=F["body"], fill=(*MUTED, alpha))
    px = x + 26 + int(text_width(draw, prefix, F["body"]))
    draw.text((px, line_y - 2), answer, font=F["body_b"], fill=(*color, alpha))
    ax = px + int(text_width(draw, answer, F["body_b"]))
    draw.text((ax, line_y), suffix, font=F["body"], fill=(*MUTED, alpha))
    draw.ellipse((W - 90 + slide, y + 30, W - 54 + slide, y + 66), fill=(*color, min(alpha, 255)))
    draw.text((W - 82 + slide, y + 30), "✓", font=F["small_b"], fill=(5, 24, 14, alpha))


def scene_three(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    centered(draw, 138, "同じ「しこう」。", F["large"], WHITE)
    centered(draw, 210, "答えは、文脈で変わる。", F["large"], WHITE)
    context_card(draw, 350, "法律", "法律を", "施行", "する", GREEN, span(t, 0.45, 0.45))
    context_card(draw, 550, "検証", "環境で", "試行", "する", CYAN, span(t, 1.55, 0.45))
    context_card(draw, 750, "通信", "アンテナの", "指向", "性", YELLOW, span(t, 2.65, 0.45))

    if t > 3.35:
        p = ease(span(t, 3.35, 0.45))
        round_rect(draw, (70, 1000, W - 70, 1115), 24, (*GREEN, int(30 * p)), (*GREEN, int(120 * p)), 2)
        centered(draw, 1022, "文字列ではなく、意味を選ぶ。", F["mid"], (*WHITE, int(255 * p)))
        centered(draw, 1152, "すべてローカルの実モデル判定", F["small"], (*MUTED, int(255 * p)))


def scene_melody(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    label(draw, 36, 112, "HARD CASE", RED)
    draw.text((36, 180), "区切り方まで、", font=F["large"], fill=WHITE)
    draw.text((36, 248), "AIが読み直す。", font=F["large"], fill=WHITE)
    card_y = 385
    round_rect(draw, (36, card_y, W - 36, card_y + 215), 26, PANEL, (54, 66, 77), 1)
    draw.text((66, card_y + 30), "この曲の", font=F["body"], fill=MUTED)
    draw.text((66, card_y + 88), "しゅせんりつ", font=F["mid"], fill=WHITE)
    draw.text((332, card_y + 96), "は、サビを支える。", font=F["small"], fill=MUTED)
    draw.line((66, card_y + 143, 292, card_y + 143), fill=GREEN, width=4)

    if t < 2.0:
        p = ease(span(t, 0.65, 0.35))
        if p:
            y = 690
            centered(draw, y, "従来候補", F["small_b"], RED)
            x = 177
            round_rect(draw, (x, y + 58, x + 172, y + 120), 16, (*RED, 30), (*RED, 120), 2)
            draw.text((x + 25, y + 70), "主戦", font=F["mid"], fill=RED)
            draw.text((x + 185, y + 70), "＋", font=F["mid"], fill=MUTED)
            round_rect(draw, (x + 250, y + 58, x + 354, y + 120), 16, (*RED, 30), (*RED, 120), 2)
            draw.text((x + 283, y + 70), "率", font=F["mid"], fill=RED)
            centered(draw, y + 155, "意味の途中で分かれてしまう", F["small"], MUTED)
    else:
        p = ease(span(t, 2.0, 0.42))
        draw_glow_circle(image, (W // 2, 795), int(150 * p), GREEN, int(50 * p))
        centered(draw, 690, "Yamatana AI IME", F["small_b"], GREEN)
        box_w = int(330 * p)
        round_rect(draw, (W // 2 - box_w // 2, 748, W // 2 + box_w // 2, 830), 20, (*GREEN, 34), (*GREEN, 180), 2)
        if p > 0.55:
            centered(draw, 756, "主旋律", F["large"], GREEN)
        centered(draw, 875, "再結合 → 再候補化 → AIで選択", F["body_b"], WHITE)
        centered(draw, 936, "誤分節をまたいで、正解へ。", F["mid"], WHITE)
        label(draw, 205, 1056, "2nd  →  1st", GREEN, 310)


def draw_chip(draw: ImageDraw.ImageDraw, cx: int, cy: int, phase: float) -> None:
    glow = int(30 + 18 * pulse(phase, 1.3))
    for inset, color in [(0, (48, 61, 72)), (12, (9, 19, 16)), (28, (13, 43, 27))]:
        round_rect(draw, (cx - 150 + inset, cy - 115 + inset, cx + 150 - inset, cy + 115 - inset), 28 - inset // 3, color, (*GREEN, 80), 2)
    draw.text((cx - 89, cy - 49), "LOCAL", font=F["latin_mid"], fill=GREEN)
    draw.text((cx - 57, cy + 4), "AI", font=F["hero"], fill=WHITE)
    for i in range(5):
        y = cy - 80 + i * 40
        draw.line((cx - 180, y, cx - 150, y), fill=(*GREEN,), width=4)
        draw.line((cx + 150, y, cx + 180, y), fill=(*GREEN,), width=4)
    draw.ellipse((cx + 116, cy - 101, cx + 135, cy - 82), fill=(0, min(255, 190 + glow), 82))


def scene_privacy(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    centered(draw, 150, "その文章、", F["large"], WHITE)
    centered(draw, 220, "外には出しません。", F["large"], WHITE)
    draw_chip(draw, W // 2, 560, t)

    # Sentences flow into the chip and stop there.
    p = smooth(span(t, 0.3, 1.3))
    x = int(-430 + p * 680)
    round_rect(draw, (x, 355, x + 390, 410), 18, PANEL_2, (70, 83, 95), 1)
    draw.text((x + 18, 367), "改正法は来月から施行…", font=F["small"], fill=MUTED)
    if t > 1.45:
        lock_y = 803
        draw.arc((W // 2 - 42, lock_y - 48, W // 2 + 42, lock_y + 34), 180, 360, fill=GREEN, width=8)
        round_rect(draw, (W // 2 - 66, lock_y, W // 2 + 66, lock_y + 105), 18, (*GREEN, 40), (*GREEN, 180), 3)
        draw.ellipse((W // 2 - 9, lock_y + 30, W // 2 + 9, lock_y + 48), fill=GREEN)
        draw.rectangle((W // 2 - 5, lock_y + 43, W // 2 + 5, lock_y + 74), fill=GREEN)
    if t > 2.0:
        centered(draw, 945, "推論・辞書・文脈処理", F["body"], MUTED)
        centered(draw, 997, "すべて、このPCの中。", F["mid"], WHITE)
        label(draw, 150, 1090, "NO CLOUD  •  NO TELEMETRY", GREEN, 420)


def draw_big_logo(image: Image.Image, cx: int, cy: int, p: float) -> None:
    radius = int(78 * p)
    draw_glow_circle(image, (cx, cy), int(125 * p), GREEN, int(70 * p))
    d = ImageDraw.Draw(image)
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=GREEN)
    bolt = [
        (cx - 8, cy - 57),
        (cx - 36, cy + 1),
        (cx - 4, cy + 1),
        (cx - 17, cy + 57),
        (cx + 42, cy - 13),
        (cx + 10, cy - 13),
    ]
    d.polygon(bolt, fill=(0, 94, 44))


def scene_end(image: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(image)
    p = ease(span(t, 0.05, 0.6))
    draw_big_logo(image, W // 2, 285, p)
    centered(draw, 420, "Yamatana", F["hero"], WHITE)
    centered(draw, 510, "AI IME", F["hero"], GREEN)
    centered(draw, 640, "変換するだけのIMEから、", F["mid"], WHITE)
    centered(draw, 700, "文脈を読むIMEへ。", F["mid"], WHITE)

    y = 822
    chips = [("Windows", CYAN), ("ローカルAI", GREEN), ("Beta", YELLOW)]
    widths = [int(text_width(draw, text, F["small_b"])) + 34 for text, _ in chips]
    x = (W - sum(widths) - 20 * (len(chips) - 1)) // 2
    for (text, color), width in zip(chips, widths):
        round_rect(draw, (x, y, x + width, y + 48), 22, (*color, 28), (*color, 110), 1)
        draw.text((x + 17, y + 9), text, font=F["small_b"], fill=color)
        x += width + 20

    q = ease(span(t, 1.0, 0.5))
    if q:
        round_rect(draw, (68, 954, W - 68, 1043), 26, (*GREEN, int(40 * q)), (*GREEN, int(170 * q)), 2)
        centered(draw, 975, "GitHubで公開中", F["mid"], (*WHITE, int(255 * q)))
        centered(draw, 1082, "github.com/YAMA-TANA/fumiori-ai-ime", F["small_b"], (*MUTED, int(255 * q)))
    draw.text((36, 1218), "YAMATANA AI IME  /  LOCAL-FIRST JAPANESE INPUT", font=F["tiny"], fill=(88, 101, 113))


def render_frame(t: float) -> Image.Image:
    image = base_frame(t)
    if t < 2.6:
        scene_hook(image, t)
    elif t < 8.0:
        scene_legal(image, t - 2.6)
    elif t < 13.9:
        scene_three(image, t - 8.0)
    elif t < 18.2:
        scene_melody(image, t - 13.9)
    elif t < 21.8:
        scene_privacy(image, t - 18.2)
    else:
        scene_end(image, t - 21.8)

    # Scene-edge flash accents.
    for edge in (2.6, 8.0, 13.9, 18.2, 21.8):
        distance = abs(t - edge)
        if distance < 0.14:
            a = int(90 * (1 - distance / 0.14))
            overlay = Image.new("RGBA", (W, H), (0, 208, 82, a))
            image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")


def add_tone(audio: np.ndarray, start: float, duration: float, freq: float, amp: float, sr: int, pan: float = 0.0) -> None:
    i0 = max(0, int(start * sr))
    i1 = min(len(audio), int((start + duration) * sr))
    if i1 <= i0:
        return
    tt = np.arange(i1 - i0) / sr
    attack = np.minimum(1.0, tt / 0.02)
    release = np.minimum(1.0, (duration - tt) / 0.12)
    env = np.clip(attack * release, 0, 1)
    tone = (np.sin(math.tau * freq * tt) + 0.22 * np.sin(math.tau * freq * 2 * tt)) * env * amp
    audio[i0:i1, 0] += tone * (1 - pan * 0.45)
    audio[i0:i1, 1] += tone * (1 + pan * 0.45)


def make_audio(path: Path) -> None:
    sr = 48_000
    samples = int(DURATION * sr)
    audio = np.zeros((samples, 2), dtype=np.float32)
    # Quiet, original electronic bed.
    chord_roots = [110.0, 130.81, 98.0, 146.83]
    for bar_start in np.arange(0, DURATION, 2.0):
        root = chord_roots[int(bar_start / 2) % len(chord_roots)]
        for ratio, amp in [(1.0, 0.025), (1.5, 0.016), (2.0, 0.012)]:
            add_tone(audio, float(bar_start), 2.15, root * ratio, amp, sr)
    # Soft rhythmic pulse.
    for beat in np.arange(0.3, DURATION, 0.5):
        i0 = int(beat * sr)
        n = min(int(0.11 * sr), samples - i0)
        if n <= 0:
            continue
        tt = np.arange(n) / sr
        kick = np.sin(math.tau * (88 - 42 * tt) * tt) * np.exp(-tt * 28) * 0.10
        audio[i0:i0+n] += kick[:, None]
    # Conversion confirmations and transitions.
    for event in [5.8, 8.0, 9.1, 10.2, 11.3, 15.9, 18.2, 21.8]:
        add_tone(audio, event, 0.24, 660, 0.075, sr, -0.2)
        add_tone(audio, event + 0.035, 0.30, 990, 0.055, sr, 0.2)
    # Tiny typing ticks.
    rng = np.random.default_rng(20260906)
    for event in [3.25, 3.40, 3.55, 3.70, 3.85, 4.00]:
        i0 = int(event * sr)
        n = int(0.025 * sr)
        noise = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (sr * 0.004)) * 0.035
        audio[i0:i0+n] += noise[:, None]
    peak = float(np.max(np.abs(audio)))
    if peak > 0.94:
        audio *= 0.94 / peak
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_video() -> Path:
    audio_path = OUT / "soundtrack.wav"
    silent_path = OUT / "yamatana_ai_ime_pr_silent.mp4"
    final_path = OUT / "yamatana_ai_ime_pr_vertical.mp4"
    poster_path = OUT / "poster.png"
    make_audio(audio_path)

    ffmpeg = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(silent_path),
    ]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    assert process.stdin is not None
    total_frames = int(DURATION * FPS)
    for index in range(total_frames):
        frame = render_frame(index / FPS)
        if index == int(22.75 * FPS):
            frame.save(poster_path, quality=95)
        process.stdin.write(frame.tobytes())
        if index % 90 == 0:
            print(f"rendered {index}/{total_frames}", flush=True)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg video encoding failed")

    mux = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(silent_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", str(final_path),
    ]
    subprocess.run(mux, check=True)
    return final_path


if __name__ == "__main__":
    print(make_video())
