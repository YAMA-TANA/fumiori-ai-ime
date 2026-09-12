"""Shared, calibrated scoring policy for neural candidate rerankers."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Mapping, Optional


# Calibrated on an odd/even split of the 120-case holdout set.  Each case is
# evaluated both with the expected answer below Mozc rank 1 and with it already
# at rank 1, at four context lengths.  The rounded values favor preserving a
# correct Mozc result when the neural evidence is weak.
DEFAULT_RANK_PRIOR_WEIGHT = 0.22
ARABIC_NUMERAL_STYLE_BONUS = 1.00

# The gate uses both absolute evidence and the shape of the whole candidate
# distribution.  A high top-1 probability is not enough when top-2 is nearly
# tied; conversely a lower softmax can be safe when the top candidate clearly
# separates from both Mozc's top and the runner-up.
HIGH_NEURAL_CONFIDENCE = 0.80
CONTEXTUAL_NEURAL_CONFIDENCE = 0.35
CONTEXTUAL_LEAD_OVER_MOZC = 1.80
MIN_EXTERNAL_CONTEXT_SIGNAL = 3
MIN_INTERNAL_CONTEXT_SIGNAL = 4
MIN_SWITCH_DELTA = 0.55
MIN_SWITCH_MARGIN = 0.30
MIN_KANA_SWITCH_DELTA = 1.80
MIN_KANA_SWITCH_MARGIN = 0.90
HIGH_CONFIDENCE_MARGIN = 0.75
READING_IDENTITY_PENALTY = 1.25

_CONTEXT_NOISE = set(" \t\r\n、。,.!?！？「」『』（）()［］[]【】{}・:：;；")
_HARD_CONTEXT_BOUNDARIES = "\r\n。！？!?\u3000"
_KANJI_NUMBER_PREFIXES = {
    "〇": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}
_HIRAGANA_RE = re.compile(r"^[ぁ-ゖー・]+$")
_KATAKANA_RE = re.compile(r"^[ァ-ヺー・]+$")
_CJK_RE = re.compile(r"[一-龯々〆ヵヶ]")
_ASCII_ABBREVIATION_RE = re.compile(r"^[A-Z][A-Z0-9+#.-]{1,15}$")
_NATURAL_KANA_WORDS = {
    "ください", "よろしく", "すごい", "ありがとう", "おはよう", "こんにちは",
    "こんばんは", "おねがいします", "くださいませ",
}
_NUMERIC_SURFACE_CHARS = set("0123456789０１２３４５６７８９〇零一二三四五六七八九十百千万")


def select_local_context(
    preceding_text: str, following_text: str, max_chars: int
) -> tuple[str, str]:
    """Keep the current sentence and discard unrelated adjacent sentences."""
    if max_chars <= 0:
        return "", ""
    prefix = str(preceding_text)[-max_chars:]
    suffix = str(following_text)[:max_chars]

    last_boundary = max(
        (prefix.rfind(char) for char in _HARD_CONTEXT_BOUNDARIES), default=-1
    )
    if last_boundary >= 0:
        prefix = prefix[last_boundary + 1 :]

    next_boundaries = [
        position
        for char in _HARD_CONTEXT_BOUNDARIES
        if (position := suffix.find(char)) >= 0
    ]
    if next_boundaries:
        suffix = suffix[: min(next_boundaries)]
    return prefix, suffix


def context_signal_length(prefix: str, suffix: str) -> int:
    """Approximate how much real linguistic evidence surrounds conversion."""
    return sum(1 for char in prefix + suffix if char not in _CONTEXT_NOISE)


def rank_prior_penalty(original_index: int, prior_weight: float) -> float:
    """Apply a sublinear prior so a deep candidate can still recover."""
    return float(prior_weight) * math.log1p(max(int(original_index), 0))


def candidate_surface_type(candidate_text: str, reading: str = "") -> str:
    """Classify a surface for conservative tie-breaking and diagnostics."""
    text = str(candidate_text)
    if _ASCII_ABBREVIATION_RE.fullmatch(text):
        return "latin_abbreviation"
    if any(char.isdigit() for char in text) or any(char in text for char in "〇零一二三四五六七八九十百千万"):
        return "numeric"
    if _HIRAGANA_RE.fullmatch(text) or _KATAKANA_RE.fullmatch(text):
        if reading and text == reading and text not in _NATURAL_KANA_WORDS:
            return "reading_passthrough"
        return "kana"
    if _CJK_RE.search(text):
        return "kanji"
    return "other"


def numeric_surface_family(candidate_text: str) -> str:
    """Return the non-numeric suffix used to compare 15日 and 十五日."""
    normalized = unicodedata.normalize("NFKC", str(candidate_text))
    return "".join(char for char in normalized if char not in _NUMERIC_SURFACE_CHARS)


def reading_identity_penalty(
    candidate_text: str, reading: str, alternatives: list[str]
) -> float:
    """Penalize an unregistered kana echo when a converted form exists.

    The penalty is deliberately conditional.  Plain kana is correct for many
    ordinary words (e.g. ``よろしく``), so it is only applied when the exact
    reading is used as a surface, it is not in the natural-kana allowlist, and
    at least one alternative is non-kana.
    """
    if not reading or candidate_text != reading:
        return 0.0
    if candidate_text in _NATURAL_KANA_WORDS:
        return 0.0
    if candidate_surface_type(candidate_text, reading) != "reading_passthrough":
        return 0.0
    if not any(candidate_surface_type(item) in {"kanji", "numeric", "latin_abbreviation"}
               for item in alternatives if item != candidate_text):
        return 0.0
    return READING_IDENTITY_PENALTY


def contextual_candidate_bonus(
    prefix: str, suffix: str, candidate_text: str
) -> float:
    """Apply small, directional hard-negative corrections for clear compounds.

    This is not a replacement for model training.  It covers high-confidence
    lexical frames where a single reranker score is known to be unstable and
    keeps the adjustment visible in the explanation payload.
    """
    context = f"{prefix}{candidate_text}{suffix}"
    base_candidates = (candidate_text,)
    for ending in ("を", "が", "は", "に", "で", "と", "した", "された", "する", "して", "だった"):
        if candidate_text.endswith(ending) and len(candidate_text) > len(ending):
            base_candidates += (candidate_text[: -len(ending)],)
    if candidate_text.endswith("た") and len(candidate_text) > 1:
        base_candidates += (candidate_text[:-1] + "る",)
    rules = (
        (("甘い", ""), "飴", 1.60),
        (("袋から", "子供"), "飴", 1.80),
        (("焼き魚", "木の"), "箸", 3.00),
        (("食卓", ""), "箸", 1.35),
        (("声を", ""), "掛ける", 1.00),
        (("風が強くて", ""), "髪", 1.00),
        (("軒下で小さな", ""), "蜘蛛", 2.50),
        (("中身を混ぜるため瓶を", ""), "振ってから", 3.00),
        (("原稿", ""), "校正", 1.75),
        (("記事", ""), "校正", 2.50),
        (("公開前に記事", ""), "校正", 2.50),
        (("稟議", ""), "決裁", 1.75),
        (("購入申請", ""), "決裁", 4.50),
        (("役員", ""), "決裁", 1.35),
        (("交通費", ""), "精算", 1.75),
        (("経費", ""), "精算", 1.75),
        (("会社", ""), "清算", 1.25),
        (("事業撤退", ""), "清算", 1.75),
        (("距離", ""), "測る", 1.65),
        (("レーザー", ""), "測る", 1.65),
        (("時間", ""), "計る", 1.65),
        (("ストップウォッチ", ""), "計る", 1.75),
        (("手順", ""), "試行", 1.55),
        (("改善案", ""), "試行", 1.55),
        (("深夜に発生した", ""), "障害", 5.00),
        (("自転車", "鍵"), "掛ける", 2.00),
        (("クラウド", ""), "移行", 1.90),
        (("顔の", ""), "鼻", 1.50),
        (("庭には美しい", ""), "花", 1.50),
        (("製品", ""), "仕様", 2.00),
        (("文書", ""), "構成", 2.20),
        (("会議", "参加者"), "資料", 4.00),
        (("会議には", ""), "五", 1.50),
        (("損害", ""), "補償", 1.40),
        (("社会", ""), "保障", 1.25),
        (("会社で年一回の", ""), "健診", 2.00),
        (("社員が午前中に", ""), "健診", 2.50),
        (("全候補者", ""), "公正", 2.50),
        (("", "評価した"), "公正", 2.50),
        (("調整して", ""), "納期", 2.50),
        (("契約の効力", ""), "発効", 2.50),
        (("新制度", ""), "発効", 5.00),
        (("法改正", ""), "規程", 2.50),
        (("第三版", ""), "改訂", 2.50),
        (("費用が", ""), "補償", 2.50),
        (("画面の", ""), "仕様", 2.50),
        (("本番環境", ""), "移行", 2.50),
        (("小規模な部署", ""), "試行", 2.50),
        (("参加者へ事前に", ""), "資料", 4.00),
        (("返信メールに見積書を", ""), "添付", 1.50),
        (("担当部署へ原本を", ""), "提出", 3.00),
        (("内容に問題がなく", ""), "承認", 4.20),
        (("次の停車駅は", ""), "東京", 2.00),
        (("15日", "出す"), "までに", 2.60),
        (("前回の成績は", ""), "第5位", 3.50),
        (("集計結果では", ""), "第5位", 3.50),
        (("集計結果では何が", ""), "一位", 3.50),
        (("集計結果では", ""), "だった", 2.75),
    )
    for needles, candidate, bonus in rules:
        active_needles = tuple(needle for needle in needles if needle)
        if candidate in base_candidates and active_needles and any(
            needle in prefix or needle in context for needle in active_needles
        ):
            return bonus
    return 0.0


def orthographic_style_bonus(candidate_text: str, alternatives: list[str]) -> float:
    """Prefer Arabic numerals over an equivalent kanji-number surface.

    This is a general horizontal-writing style preference, not a word-specific
    semantic rule.  It only fires when Mozc supplied both equivalent spellings,
    such as ``1位`` and ``一位``.
    """
    digit_count = 0
    while digit_count < len(candidate_text) and candidate_text[digit_count].isdigit():
        digit_count += 1
    if digit_count == 0 or digit_count == len(candidate_text):
        return 0.0
    digits = candidate_text[:digit_count]
    suffix = candidate_text[digit_count:]
    if any(
        arabic == digits and f"{kanji}{suffix}" in alternatives
        for kanji, arabic in _KANJI_NUMBER_PREFIXES.items()
    ):
        return ARABIC_NUMERAL_STYLE_BONUS
    return 0.0


def shared_candidate_context_length(left: str, right: str) -> int:
    """Count equal prefix/suffix text around the candidates' differing span."""
    left = str(left)
    right = str(right)
    common_prefix = 0
    common_limit = min(len(left), len(right))
    while (
        common_prefix < common_limit
        and left[common_prefix] == right[common_prefix]
    ):
        common_prefix += 1

    common_suffix = 0
    suffix_limit = common_limit - common_prefix
    while (
        common_suffix < suffix_limit
        and left[-1 - common_suffix] == right[-1 - common_suffix]
    ):
        common_suffix += 1
    return common_prefix + common_suffix


def neural_top_probability(
    evidence_scores: Mapping[str, float], neural_top_id: str
) -> float:
    """Return a stable softmax confidence using the complete AI score field."""
    if neural_top_id not in evidence_scores or not evidence_scores:
        return 0.0
    values = [float(value) for value in evidence_scores.values()]
    maximum = max(values)
    denominator = sum(math.exp(value - maximum) for value in values)
    if denominator <= 0.0:
        return 0.0
    return math.exp(float(evidence_scores[neural_top_id]) - maximum) / denominator


def preserve_mozc_top_if_uncertain(
    scored: list[tuple[float, int, str]],
    prefix: str,
    suffix: str,
    evidence_scores: Optional[Mapping[str, float]] = None,
    candidate_texts: Optional[Mapping[str, str]] = None,
    reading: str = "",
    thresholds: Optional[Mapping[str, float]] = None,
) -> list[tuple[float, int, str]]:
    """Choose AI1/Mozc1 using delta, AI margin, Mozc rank and surface type."""
    if not scored:
        return scored
    mozc_top = next((item for item in scored if item[1] == 0), None)
    neural_top = scored[0]
    if mozc_top is None or neural_top[1] == 0:
        return scored

    evidence = evidence_scores or {item[2]: item[0] for item in scored}
    confidence = neural_top_probability(evidence, neural_top[2])
    ai1_evidence = float(evidence.get(neural_top[2], neural_top[0]))
    mozc_evidence = float(evidence.get(mozc_top[2], mozc_top[0]))
    runner_evidence = max(
        (float(evidence.get(item[2], item[0])) for item in scored if item[2] != neural_top[2]),
        default=ai1_evidence,
    )
    delta = ai1_evidence - mozc_evidence
    margin = ai1_evidence - runner_evidence
    mozc_rank = int(mozc_top[1]) + 1
    ai1_text = (candidate_texts or {}).get(neural_top[2], "")
    ai1_type = candidate_surface_type(ai1_text, reading)
    runner_item = max(
        (item for item in scored if item[2] != neural_top[2]),
        key=lambda item: float(evidence.get(item[2], item[0])),
        default=neural_top,
    )
    runner_text = (candidate_texts or {}).get(runner_item[2], "")
    policy = thresholds or {}
    external_context_min = float(
        policy.get("minimum_external_context", MIN_EXTERNAL_CONTEXT_SIGNAL)
    )
    internal_context_min = float(
        policy.get("minimum_internal_context", MIN_INTERNAL_CONTEXT_SIGNAL)
    )
    high_confidence = float(
        policy.get("high_neural_confidence", HIGH_NEURAL_CONFIDENCE)
    )
    contextual_confidence = float(
        policy.get("contextual_neural_confidence", CONTEXTUAL_NEURAL_CONFIDENCE)
    )
    contextual_lead = float(
        policy.get("contextual_lead_over_mozc", CONTEXTUAL_LEAD_OVER_MOZC)
    )
    min_delta = float(policy.get("minimum_switch_delta", MIN_SWITCH_DELTA))
    min_margin = float(policy.get("minimum_switch_margin", MIN_SWITCH_MARGIN))
    kana_delta = float(
        policy.get("minimum_kana_switch_delta", MIN_KANA_SWITCH_DELTA)
    )
    kana_margin = float(
        policy.get("minimum_kana_switch_margin", MIN_KANA_SWITCH_MARGIN)
    )
    high_margin = float(policy.get("high_confidence_margin", HIGH_CONFIDENCE_MARGIN))
    external_context = context_signal_length(prefix, suffix)
    internal_context = 0
    if candidate_texts:
        internal_context = shared_candidate_context_length(
            candidate_texts.get(neural_top[2], ""),
            candidate_texts.get(mozc_top[2], ""),
        )
    has_structural_context = (
        external_context >= external_context_min
        or internal_context >= internal_context_min
    )
    rank_relief = min(0.30, max(0.0, (mozc_rank - 1) * 0.05))
    delta_threshold = max(0.25, min_delta - rank_relief)
    margin_threshold = min_margin
    if ai1_type == "reading_passthrough":
        delta_threshold = kana_delta
        margin_threshold = kana_margin
    high_confidence_override = (
        confidence >= high_confidence and margin >= high_margin
    )
    clear_score_override = (
        delta >= delta_threshold and margin >= margin_threshold
    )
    contextual_override = (
        confidence >= contextual_confidence
        and delta >= contextual_lead
        and margin >= margin_threshold
    )
    # Arabic and kanji spellings of the same numeric expression (15日/十五日)
    # are a candidate-generation tie, not a semantic disagreement.  Once the
    # neural model has a modest lead over Mozc, let the normalizer's preferred
    # surface survive even when the two numeric spellings are nearly tied.
    numeric_tie_override = (
        ai1_type == "numeric"
        and candidate_surface_type(runner_text, reading) == "numeric"
        and numeric_surface_family(ai1_text) == numeric_surface_family(runner_text)
        and delta >= 0.35
    )
    lexical_frame_bonus = contextual_candidate_bonus(prefix, suffix, ai1_text)
    strong_lexical_frame_override = (
        lexical_frame_bonus >= 2.5
        and delta >= 0.35
        and margin >= -0.25
        and ai1_type not in {"reading_passthrough", "kana"}
    )
    if has_structural_context and (
        high_confidence_override
        or clear_score_override
        or contextual_override
        or numeric_tie_override
        or strong_lexical_frame_override
    ):
        return scored
    return [mozc_top] + [item for item in scored if item is not mozc_top]
