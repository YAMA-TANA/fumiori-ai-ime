"""Self-contained ONNX Runtime backend for Yamatana AI IME."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from product_settings import domain_instruction, load_settings, normalize_settings
from ranker.lexicon import LexicalKnowledge
from ranker.scoring import (
    ARABIC_NUMERAL_STYLE_BONUS,
    CONTEXTUAL_LEAD_OVER_MOZC,
    CONTEXTUAL_NEURAL_CONFIDENCE,
    DEFAULT_RANK_PRIOR_WEIGHT,
    HIGH_CONFIDENCE_MARGIN,
    HIGH_NEURAL_CONFIDENCE,
    MIN_EXTERNAL_CONTEXT_SIGNAL,
    MIN_INTERNAL_CONTEXT_SIGNAL,
    MIN_KANA_SWITCH_DELTA,
    MIN_KANA_SWITCH_MARGIN,
    MIN_SWITCH_DELTA,
    MIN_SWITCH_MARGIN,
    context_signal_length as _context_signal_length,
    contextual_candidate_bonus,
    reading_identity_penalty,
    neural_top_probability as _neural_top_probability,
    orthographic_style_bonus,
    preserve_mozc_top_if_uncertain as _preserve_mozc_top_if_uncertain,
    rank_prior_penalty as _rank_prior_penalty,
    select_local_context,
    shared_candidate_context_length as _shared_candidate_context_length,
)


LOG = logging.getLogger("yamatana_ai_ime.onnx_ranker")


def _runtime_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(getattr(sys, "_MEIPASS")))
    roots.extend((Path(sys.executable).parent, Path(__file__).resolve().parents[1]))
    return roots


def _resolve_first(relative_paths: tuple[str, ...]) -> Optional[Path]:
    for root in _runtime_roots():
        for relative in relative_paths:
            candidate = root / relative
            if candidate.exists():
                return candidate
    return None


class OnnxRuriReranker:
    """Ruri IME reranker without a Python, PyTorch, or CUDA prerequisite."""

    def __init__(
        self,
        settings: Optional[Mapping[str, Any]] = None,
        settings_path: Optional[str | Path] = None,
        prior_w: float = DEFAULT_RANK_PRIOR_WEIGHT,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.settings = normalize_settings(settings) if settings is not None else load_settings(settings_path)
        self.prior_w = float(prior_w)
        self.context_enabled = bool(self.settings["context_enabled"])
        self.context_chars = int(self.settings["context_chars"])
        self.document_domain = str(self.settings["document_domain"])
        self.document_instruction = domain_instruction(self.settings)
        self.enable_lexical_grounding = bool(self.settings["lexical_grounding"])
        self.lexicon = LexicalKnowledge() if self.enable_lexical_grounding else None

        available = set(ort.get_available_providers())
        requested = str(self.settings["compute_mode"])
        use_gpu = requested in {"auto", "gpu"} and "DmlExecutionProvider" in available
        if requested == "gpu" and not use_gpu:
            raise RuntimeError(
                "GPU演算が選択されていますがDirectML対応GPUを利用できません。"
                "設定で「自動選択」または「CPU」を選んでください。"
            )

        if model_path is not None:
            resolved_model = Path(model_path)
        elif use_gpu:
            resolved_model = _resolve_first((
                "build/onnx-model-70m/ruri-ime-fp16.onnx",
                "models/onnx/ruri-ime-fp16.onnx",
                "build/onnx-model/ruri-ime-fp16.onnx",
                "models/onnx/ruri-ime-fp32.onnx",
            ))
        else:
            resolved_model = _resolve_first((
                "build/onnx-model-70m/ruri-ime-int8.onnx",
                "models/onnx/ruri-ime-int8.onnx",
                "build/onnx-model/ruri-ime-int8.onnx",
                "models/onnx/ruri-ime-fp32.onnx",
            ))
        if not resolved_model or not resolved_model.exists():
            raise FileNotFoundError("配布用ONNXモデルが見つかりません。再インストールしてください。")
        tokenizer_path = _resolve_first((
            "build/onnx-model-70m/tokenizer.json",
            "models/onnx/tokenizer.json",
            "models/ruri-v3-70m-ime-distilled/tokenizer.json",
            "models/ruri-v3-reranker-310m-ime-tuned/tokenizer.json",
        ))
        if tokenizer_path is None:
            raise FileNotFoundError("AI tokenizer.json が見つかりません。再インストールしてください。")

        self.model_path = str(resolved_model.resolve())
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=3, pad_token="<pad>")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = min(8, max(2, os.cpu_count() or 2))
        options.inter_op_num_threads = 1
        if use_gpu:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        LOG.info("loading %s with requested providers=%s", resolved_model, providers)
        self.session = ort.InferenceSession(
            str(resolved_model), sess_options=options, providers=providers
        )
        active_providers = list(self.session.get_providers())
        directml_active = "DmlExecutionProvider" in active_providers
        self.device = "gpu-directml" if directml_active else "cpu"
        if requested == "gpu" and not directml_active:
            raise RuntimeError(
                "GPU演算を要求しましたが、ONNX RuntimeセッションでDirectMLが有効になりませんでした。"
            )
        if use_gpu and not directml_active:
            LOG.warning(
                "DirectML was requested but the active session providers are %s; using CPU",
                active_providers,
            )
        else:
            LOG.info("active ONNX providers=%s device=%s", active_providers, self.device)
        self._warmup()

    def _encode(self, queries: list[str], documents: list[str]) -> dict[str, np.ndarray]:
        encodings = self.tokenizer.encode_batch(list(zip(queries, documents)))
        return {
            "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            ),
        }

    def _warmup(self) -> None:
        # Production reranking normally scores four Mozc surfaces.  Warming the
        # same shape avoids compiling a much larger 32-row DirectML graph that
        # the interactive path never uses.
        warmup_count = 4
        query = "文書方針: 一般的な日本語文書。\n文脈に合う表記を選びなさい。"
        inputs = self._encode(
            [query] * warmup_count,
            [f"文章の変換候補{index}" for index in range(warmup_count)],
        )
        self.session.run(["logits"], inputs)

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        request_id = request.get("request_id", 0)
        prefix = str(request.get("preceding_text", request.get("prefix", "")))
        suffix = str(request.get("following_text", request.get("suffix", "")))
        reading = str(request.get("read", request.get("reading", "")))
        candidates = request.get("candidates", [])
        if not candidates:
            return {"request_id": request_id, "candidates": []}

        if not self.context_enabled or self.context_chars <= 0:
            prefix = ""
            suffix = ""
        else:
            prefix, suffix = select_local_context(
                prefix, suffix, self.context_chars
            )

        all_candidates = []
        for index, candidate in enumerate(candidates):
            word = str(candidate.get("text", candidate.get("word", "")))
            candidate_id = str(candidate.get("id", f"c{index + 1}"))
            all_candidates.append((index, candidate_id, word))

        # The candidate document already contains the local context.  Repeating
        # the same prefix/suffix inside every query roughly doubles the token
        # work, so keep the query short and candidate-independent.
        query = (
            f"文書方針: {self.document_instruction}\n"
            f"読み「{reading}」の変換候補として、文脈に最も適切な表記を選びなさい。"
        )
        encode_started = time.perf_counter()
        inputs = self._encode(
            [query] * len(all_candidates),
            [f"{prefix}{word}{suffix}" for _index, _candidate_id, word in all_candidates],
        )
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        inference_started = time.perf_counter()
        logits = np.asarray(self.session.run(["logits"], inputs)[0]).reshape(-1)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        scored: list[tuple[float, int, str]] = []
        evidence_scores: dict[str, float] = {}
        score_details: dict[str, dict[str, float | int | str]] = {}
        candidate_texts = [word for _index, _candidate_id, word in all_candidates]
        for original_index, candidate_id, word in all_candidates:
            raw_score = float(logits[original_index])
            lexical_penalty = (
                self.lexicon.compute_lexical_penalty(word, reading)
                if self.lexicon and reading
                else 0.0
            )
            style_bonus = orthographic_style_bonus(word, candidate_texts)
            context_bonus = contextual_candidate_bonus(prefix, suffix, word)
            reading_penalty = reading_identity_penalty(word, reading, candidate_texts)
            evidence_score = (
                raw_score + style_bonus + context_bonus
                - lexical_penalty - reading_penalty
            )
            prior_penalty = _rank_prior_penalty(original_index, self.prior_w)
            final_score = evidence_score - prior_penalty
            evidence_scores[candidate_id] = float(evidence_score)
            scored.append((float(final_score), original_index, candidate_id))
            score_details[candidate_id] = {
                "id": candidate_id,
                "text": word,
                "original_rank": original_index + 1,
                "model_score": float(raw_score),
                "style_bonus": float(style_bonus),
                "context_bonus": float(context_bonus),
                "lexical_penalty": float(lexical_penalty),
                "reading_identity_penalty": float(reading_penalty),
                "rank_prior_penalty": float(prior_penalty),
                "evidence_score": float(evidence_score),
                "final_score": float(final_score),
            }

        # Mozc rank is only a soft prior.  For exact score ties the stable
        # original order is retained, but a clear AI evidence gap can promote
        # a deep candidate without paying a linearly growing rank tax.
        scored.sort(key=lambda item: item[0], reverse=True)
        neural_top_id = scored[0][2]
        mozc_top_id = next(item[2] for item in scored if item[1] == 0)
        lead_over_mozc = None
        lead_over_runner = None
        if neural_top_id != mozc_top_id:
            lead_over_mozc = (
                evidence_scores[neural_top_id] - evidence_scores[mozc_top_id]
            )
            runner_evidence = max(
                evidence_scores[item[2]] for item in scored[1:]
            )
            lead_over_runner = evidence_scores[neural_top_id] - runner_evidence
        candidate_text_by_id = {
            candidate_id: word for _index, candidate_id, word in all_candidates
        }
        internal_context_len = _shared_candidate_context_length(
            candidate_text_by_id[neural_top_id], candidate_text_by_id[mozc_top_id]
        )
        neural_confidence = _neural_top_probability(evidence_scores, neural_top_id)
        pre_gate_scored = list(scored)
        scored = _preserve_mozc_top_if_uncertain(
            scored,
            prefix,
            suffix,
            evidence_scores,
            candidate_text_by_id,
            reading,
        )
        selected_id = scored[0][2]
        context_len = _context_signal_length(prefix, suffix)
        for output_rank, (_score, _original_index, candidate_id) in enumerate(
            scored, start=1
        ):
            score_details[candidate_id]["output_rank"] = output_rank
        self.last_explanation = {
            "context": {
                "preceding_text": prefix,
                "following_text": suffix,
                "reading": reading,
                "signal_length": context_len,
            },
            "formula": (
                "evidence = model + style_bonus + context_bonus "
                "- lexical_penalty - reading_identity_penalty; "
                "final = evidence - rank_prior_penalty"
            ),
            "parameters": {
                "rank_prior_weight": self.prior_w,
                "arabic_numeral_style_bonus": ARABIC_NUMERAL_STYLE_BONUS,
                "high_neural_confidence": HIGH_NEURAL_CONFIDENCE,
                "contextual_neural_confidence": CONTEXTUAL_NEURAL_CONFIDENCE,
                "contextual_lead_over_mozc": CONTEXTUAL_LEAD_OVER_MOZC,
                "minimum_external_context": MIN_EXTERNAL_CONTEXT_SIGNAL,
                "minimum_internal_context": MIN_INTERNAL_CONTEXT_SIGNAL,
                "minimum_switch_delta": MIN_SWITCH_DELTA,
                "minimum_switch_margin": MIN_SWITCH_MARGIN,
                "minimum_kana_switch_delta": MIN_KANA_SWITCH_DELTA,
                "minimum_kana_switch_margin": MIN_KANA_SWITCH_MARGIN,
                "high_confidence_margin": HIGH_CONFIDENCE_MARGIN,
                "reading_identity_penalty": 1.25,
            },
            "decision": {
                "neural_top_id": neural_top_id,
                "selected_id": selected_id,
                "preserved_mozc_top": neural_top_id != selected_id,
                "lead_over_mozc": lead_over_mozc,
                "lead_over_runner": lead_over_runner,
                "neural_top_probability": neural_confidence,
                "internal_context_length": internal_context_len,
                "ai1_evidence_score": float(evidence_scores[neural_top_id]),
                "mozc1_evidence_score": float(evidence_scores[mozc_top_id]),
                "ai_margin": float(
                    evidence_scores[neural_top_id]
                    - max(
                        evidence_scores[item[2]]
                        for item in pre_gate_scored[1:]
                    )
                ) if len(pre_gate_scored) > 1 else 0.0,
            },
            "candidates": sorted(
                score_details.values(), key=lambda item: int(item["output_rank"])
            ),
        }
        total_ms = (time.perf_counter() - started) * 1000.0
        self.last_latency_ms = round(total_ms, 2)
        self.last_timing = {
            "encode_ms": round(encode_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "postprocess_ms": round(max(0.0, total_ms - encode_ms - inference_ms), 3),
            "total_ms": round(total_ms, 3),
            "candidate_count": len(all_candidates),
        }
        return {
            "request_id": request_id,
            "candidates": [
                {"id": candidate_id, "rank": rank, "score": float(score)}
                for rank, (score, _original_index, candidate_id) in enumerate(scored, start=1)
            ],
        }
