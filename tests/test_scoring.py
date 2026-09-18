from ranker.scoring import (
    ARABIC_NUMERAL_STYLE_BONUS,
    CONTEXTUAL_LEAD_OVER_MOZC,
    CONTEXTUAL_NEURAL_CONFIDENCE,
    DEFAULT_RANK_PRIOR_WEIGHT,
    HIGH_NEURAL_CONFIDENCE,
    MIN_EXTERNAL_CONTEXT_SIGNAL,
    MIN_INTERNAL_CONTEXT_SIGNAL,
    MIN_SWITCH_MARGIN,
    contextual_candidate_bonus,
    neural_top_probability,
    orthographic_style_bonus,
    reading_identity_penalty,
    select_local_context,
    shared_candidate_context_length,
)


def test_local_context_discards_an_unrelated_previous_sentence():
    prefix, suffix = select_local_context(
        "彼の顔の鼻は大きい　庭には美しい", "を眺める。次の文", 128
    )
    assert prefix == "庭には美しい"
    assert suffix == "を眺める"


def test_local_context_keeps_commas_inside_the_current_sentence():
    prefix, suffix = select_local_context(
        "条件を変えて、何度も", "した結果を記録する", 128
    )
    assert prefix == "条件を変えて、何度も"
    assert suffix == "した結果を記録する"


def test_calibrated_nonlinear_policy_parameters():
    assert DEFAULT_RANK_PRIOR_WEIGHT == 0.22
    assert HIGH_NEURAL_CONFIDENCE == 0.80
    assert CONTEXTUAL_NEURAL_CONFIDENCE == 0.35
    assert CONTEXTUAL_LEAD_OVER_MOZC == 1.80
    assert MIN_EXTERNAL_CONTEXT_SIGNAL == 3
    assert MIN_INTERNAL_CONTEXT_SIGNAL == 4
    assert MIN_SWITCH_MARGIN == 0.30


def test_arabic_numeral_style_bonus_requires_an_equivalent_kanji_candidate():
    assert orthographic_style_bonus("1位", ["位置位", "一位", "1位"]) == (
        ARABIC_NUMERAL_STYLE_BONUS
    )
    assert orthographic_style_bonus("一位", ["一位", "1位"]) == 0.0
    assert orthographic_style_bonus("1位", ["位置位", "1位"]) == 0.0


def test_neural_probability_uses_all_candidate_scores():
    concentrated = neural_top_probability({"top": 2.0, "other": 0.0}, "top")
    crowded = neural_top_probability(
        {"top": 2.0, "a": 1.8, "b": 1.7, "c": 1.6}, "top"
    )
    assert concentrated > 0.8
    assert crowded < 0.4


def test_shared_candidate_context_excludes_differing_span():
    assert shared_candidate_context_length(
        "彼の顔の鼻は大きい", "彼の顔の花は大きい"
    ) == 8
    assert shared_candidate_context_length("油送して", "輸送して") == 3


def test_reading_echo_is_penalized_only_when_a_real_surface_exists():
    assert reading_identity_penalty("はなが", "はなが", ["はなが", "鼻が"]) > 0
    assert reading_identity_penalty("ください", "ください", ["ください"]) == 0


def test_directional_hard_negative_frames_cover_numeric_and_action_phrases():
    assert contextual_candidate_bonus("中身を混ぜるため瓶を", "", "振ってから") >= 3.0
    assert contextual_candidate_bonus("申請書を今月15日", "出す", "までに") >= 2.6
    assert contextual_candidate_bonus("前回の成績は", "", "第5位") >= 3.5
    assert contextual_candidate_bonus("集計結果では何が", "", "一位") >= 3.5


def test_homophone_frames_cover_measurement_and_rehabilitation_examples():
    assert contextual_candidate_bonus("月と地球の距離を", "", "測る") >= 1.65
    assert contextual_candidate_bonus("ストップウォッチで時間を", "", "計る") >= 1.75
    assert contextual_candidate_bonus("毎朝、体重を", "", "量る") >= 1.75
    assert contextual_candidate_bonus("刑務所を出たあと、指導員は彼の", "", "更生") >= 2.20
    assert contextual_candidate_bonus("私の庭の", "", "花") >= 1.25
