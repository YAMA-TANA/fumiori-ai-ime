"""Self-contained ONNX Runtime backend for Yamatana AI IME."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


_CUDA_DLL_HANDLES: list[Any] = []


def _configure_cuda_dll_search_path() -> None:
    """Make pip-installed CUDA/cuDNN DLLs visible before ORT loads providers."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates: list[Path] = []
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        bundle_root = Path(getattr(sys, "_MEIPASS"))
        candidates.extend(
            bundle_root / relative
            for relative in (
                "nvidia/cuda_runtime/bin",
                "nvidia/cuda_nvrtc/bin",
                "nvidia/cublas/bin",
                "nvidia/cudnn/bin",
            )
        )
    try:
        import importlib.util

        for package in ("nvidia.cuda_runtime", "nvidia.cuda_nvrtc", "nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(package)
            if spec and spec.submodule_search_locations:
                candidates.append(Path(next(iter(spec.submodule_search_locations))) / "bin")
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        normalized = str(candidate.resolve()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
        except OSError:
            logging.getLogger("yamatana_ai_ime.onnx_ranker").debug(
                "could not add CUDA DLL directory %s", candidate, exc_info=True
            )


import onnxruntime as ort
from tokenizers import Tokenizer

from product_settings import domain_instruction, load_settings, normalize_settings
try:
    from .protocol import validate_batch_request
except ImportError:
    from protocol import validate_batch_request
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
QUERY_VARIANTS = ("current", "short", "none")


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
    """Ruri IME reranker with CUDA, DirectML, and CPU fallback support."""

    def __init__(
        self,
        settings: Optional[Mapping[str, Any]] = None,
        settings_path: Optional[str | Path] = None,
        prior_w: Optional[float] = None,
        model_path: Optional[str | Path] = None,
        model_paths: Optional[Sequence[str | Path]] = None,
        ensemble_weights: Optional[Sequence[float]] = None,
        safety_gate: Optional[bool] = None,
        query_variant: str = "current",
        execution_providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.settings = normalize_settings(settings) if settings is not None else load_settings(settings_path)
        self.context_enabled = bool(self.settings["context_enabled"])
        self.context_chars = int(self.settings["context_chars"])
        self.document_domain = str(self.settings["document_domain"])
        self.document_instruction = domain_instruction(self.settings)
        if query_variant not in QUERY_VARIANTS:
            raise ValueError(f"unknown query variant: {query_variant}")
        self.query_variant = query_variant
        self.enable_lexical_grounding = bool(self.settings["lexical_grounding"])
        self.lexicon = LexicalKnowledge() if self.enable_lexical_grounding else None

        available = set(ort.get_available_providers())
        requested = str(self.settings["compute_mode"])
        provider_override = list(execution_providers or [])
        gpu_provider = next(
            (provider for provider in ("CUDAExecutionProvider", "DmlExecutionProvider")
             if provider in available),
            None,
        )
        use_gpu = (
            any(provider in {"DmlExecutionProvider", "CUDAExecutionProvider"}
                for provider in provider_override)
            if provider_override
            else requested in {"auto", "gpu"} and gpu_provider is not None
        )
        if requested == "gpu" and not use_gpu:
            raise RuntimeError(
                "GPU演算が選択されていますがCUDA/DirectML対応GPUを利用できません。"
                "設定で「自動選択」または「CPU」を選んでください。"
            )

        if model_paths is not None:
            if model_path is not None:
                raise ValueError("pass model_path or model_paths, not both")
            requested_models = [Path(path) for path in model_paths]
        elif model_path is not None:
            requested_models = [Path(model_path)]
        elif use_gpu:
            resolved_model = _resolve_first((
                "build/onnx-model-70m/ruri-ime-fp16.onnx",
                "models/onnx/ruri-ime-fp16.onnx",
                "build/onnx-model/ruri-ime-fp16.onnx",
                "models/onnx/ruri-ime-fp32.onnx",
            ))
            requested_models = [resolved_model] if resolved_model else []
        else:
            resolved_model = _resolve_first((
                "build/onnx-model-70m/ruri-ime-int8.onnx",
                "models/onnx/ruri-ime-int8.onnx",
                "build/onnx-model/ruri-ime-int8.onnx",
                "models/onnx/ruri-ime-fp32.onnx",
            ))
            requested_models = [resolved_model] if resolved_model else []
        if not requested_models or any(
            path is None or not path.exists() for path in requested_models
        ):
            raise FileNotFoundError("配布用ONNXモデルが見つかりません。再インストールしてください。")
        self.model_paths = [str(Path(path).resolve()) for path in requested_models]
        self.model_path = self.model_paths[0]
        # The old single-model gate is calibrated for one score distribution.
        # A calibrated multi-model ensemble already combines complementary
        # evidence, so it defaults to direct selection; callers can opt back
        # into the conservative gate with safety_gate=True.
        self.safety_gate = (
            len(self.model_paths) == 1 if safety_gate is None else bool(safety_gate)
        )
        if prior_w is None:
            self.prior_w = (
                DEFAULT_RANK_PRIOR_WEIGHT if len(self.model_paths) == 1 else 0.0
            )
        else:
            self.prior_w = float(prior_w)
        if ensemble_weights is None:
            if len(self.model_paths) == 2:
                # LoRA6 handles the preceding-only distribution better, while
                # LoRA3 supplies complementary evidence on the strict set.
                weights = [0.25, 0.75]
            else:
                weights = [1.0 / len(self.model_paths)] * len(self.model_paths)
        else:
            weights = [float(value) for value in ensemble_weights]
        if (
            len(weights) != len(self.model_paths)
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or sum(weights) <= 0.0
        ):
            raise ValueError(
                "ensemble_weights must be finite non-negative values matching model_paths"
            )
        total_weight = sum(weights)
        self.ensemble_weights = [value / total_weight for value in weights]
        self.ensemble_gate_thresholds = (
            {
                "minimum_switch_delta": 0.25,
                "minimum_switch_margin": 0.10,
                "contextual_neural_confidence": 0.25,
                "contextual_lead_over_mozc": 1.20,
            }
            if len(self.model_paths) > 1
            else None
        )
        tokenizer_path = _resolve_first((
            "build/onnx-model-70m/tokenizer.json",
            "models/onnx/tokenizer.json",
            "models/ruri-v3-70m-ime-distilled/tokenizer.json",
            "models/ruri-v3-reranker-310m-ime-tuned/tokenizer.json",
        ))
        if tokenizer_path is None:
            raise FileNotFoundError("AI tokenizer.json が見つかりません。再インストールしてください。")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=3, pad_token="<pad>")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = min(8, max(2, os.cpu_count() or 2))
        options.inter_op_num_threads = 1
        if provider_override:
            options.enable_mem_pattern = False if use_gpu else options.enable_mem_pattern
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if use_gpu else options.execution_mode
            providers = provider_override
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
        elif use_gpu:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [gpu_provider, "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers:
            _configure_cuda_dll_search_path()
            try:
                # ORT 1.21+ knows how to preload the CUDA 12/cuDNN 9 wheels on
                # Windows. Do this only when creating a CUDA session so other
                # backends (notably PyTorch) can load their own DLL versions.
                ort.preload_dlls(directory="")
            except (AttributeError, OSError):
                LOG.warning(
                    "CUDA DLL preload failed; CUDA session creation may fall back to CPU",
                    exc_info=True,
                )
        LOG.info("loading %s with requested providers=%s", self.model_paths, providers)
        self.sessions = [
            ort.InferenceSession(
                path, sess_options=options, providers=providers
            )
            for path in self.model_paths
        ]
        # Keep the singular attribute for existing diagnostics and lightweight
        # tests that inject one fake ONNX session.
        self.session = self.sessions[0]
        active_providers = list(self.session.get_providers())
        directml_active = "DmlExecutionProvider" in active_providers
        cuda_active = "CUDAExecutionProvider" in active_providers
        gpu_active = directml_active or cuda_active
        self.device = (
            "gpu-cuda" if cuda_active else
            "gpu-directml" if directml_active else "cpu"
        )
        if requested == "gpu" and not gpu_active:
            raise RuntimeError(
                "GPU演算を要求しましたが、ONNX RuntimeセッションでGPUプロバイダーが有効になりませんでした。"
            )
        if use_gpu and not gpu_active:
            LOG.warning(
                "GPU was requested but the active session providers are %s; using CPU",
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
        warmup_count = 32
        query = "文書方針: 一般的な日本語文書。\n文脈に合う表記を選びなさい。"
        inputs = self._encode(
            [query] * warmup_count,
            [f"文章の変換候補{index}" for index in range(warmup_count)],
        )
        for session in self.sessions:
            session.run(["logits"], inputs)

    def _build_query(self, prefix: str, suffix: str) -> str:
        query_variant = getattr(self, "query_variant", "current")
        if query_variant == "none":
            return ""
        if query_variant == "short":
            return f"文脈「{prefix}____{suffix}」から候補を選びなさい。"
        return (
            f"文書方針: {self.document_instruction}\n"
            f"文脈「{prefix}____{suffix}」に最も適切な表記を選びなさい。"
        )

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

        query = self._build_query(prefix, suffix)
        inputs = self._encode(
            [query] * len(all_candidates),
            [f"{prefix}{word}{suffix}" for _index, _candidate_id, word in all_candidates],
        )
        sessions = getattr(self, "sessions", [self.session])
        ensemble_weights = getattr(self, "ensemble_weights", [1.0])
        request_weights = list(ensemble_weights)
        if len(sessions) == 2 and len(reading) <= 3:
            # Very short readings have a higher semantic collision rate.  The
            # LoRA3 signal is more reliable on the rare cases where LoRA3 and
            # LoRA6 disagree, while longer practical phrases favor LoRA6.
            request_weights = [0.50, 0.50]
        logits_by_model = [
            np.asarray(session.run(["logits"], inputs)[0]).reshape(-1)
            for session in sessions
        ]
        if any(logits.shape != logits_by_model[0].shape for logits in logits_by_model[1:]):
            raise RuntimeError("ensemble models returned different candidate batch sizes")
        logits = np.average(
            np.stack(logits_by_model, axis=0),
            axis=0,
            weights=np.asarray(request_weights, dtype=np.float32),
        )

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
                "model_scores": [
                    float(model_logits[original_index])
                    for model_logits in logits_by_model
                ],
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
        if getattr(self, "safety_gate", True):
            scored = _preserve_mozc_top_if_uncertain(
                scored,
                prefix,
                suffix,
                evidence_scores,
                candidate_text_by_id,
                reading,
                getattr(self, "ensemble_gate_thresholds", None),
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
                "evidence = weighted_model_ensemble + style_bonus + context_bonus "
                "- lexical_penalty - reading_identity_penalty; "
                "final = evidence - rank_prior_penalty"
                if len(sessions) > 1 else
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
                "model_paths": list(
                    getattr(self, "model_paths", None)
                    or [getattr(self, "model_path", "")]
                ),
                "ensemble_weights": list(request_weights),
                "ensemble_gate_thresholds": getattr(
                    self, "ensemble_gate_thresholds", None
                ),
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
        self.last_latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "request_id": request_id,
            "candidates": [
                {"id": candidate_id, "rank": rank, "score": float(score)}
                for rank, (score, _original_index, candidate_id) in enumerate(scored, start=1)
            ],
        }

    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rank every conversion segment in one flat ONNX batch.

        The wire response deliberately contains only one winner and one
        confidence value per segment.  Candidate text never leaves the IME.
        Each loaded model session receives one flat ``session.run`` call; the
        two-model production ensemble therefore performs one call per model,
        rather than one call per segment.
        """
        request = validate_batch_request(request)
        started = time.perf_counter()
        context_enabled = bool(getattr(self, "context_enabled", True))
        context_chars = int(getattr(self, "context_chars", 128))

        segment_data = []
        queries: list[str] = []
        documents: list[str] = []
        for segment_index, segment in enumerate(request["segments"]):
            raw_prefix = str(segment["preceding_text"])
            raw_suffix = str(segment["following_text"])
            if not context_enabled or context_chars <= 0:
                prefix = ""
                suffix = ""
            else:
                prefix, suffix = select_local_context(
                    raw_prefix, raw_suffix, context_chars
                )

            unique_words: list[str] = []
            word_indexes: dict[str, int] = {}
            for candidate in segment["candidates"]:
                word = str(candidate["text"])
                if word not in word_indexes:
                    word_indexes[word] = len(unique_words)
                    unique_words.append(word)

            flat_indexes: dict[str, int] = {}
            query = self._build_query(prefix, suffix)
            for word in unique_words:
                flat_indexes[word] = len(queries)
                queries.append(query)
                documents.append(f"{prefix}{word}{suffix}")

            model_weights = list(getattr(self, "ensemble_weights", [1.0]))
            sessions = getattr(self, "sessions", [self.session])
            if len(model_weights) != len(sessions):
                model_weights = [1.0 / len(sessions)] * len(sessions)
            if len(sessions) == 2 and len(str(segment["read"])) <= 3:
                model_weights = [0.50, 0.50]
            segment_data.append({
                "index": segment_index,
                "id": segment["id"],
                "prefix": prefix,
                "suffix": suffix,
                "reading": str(segment["read"]),
                "candidates": segment["candidates"],
                "word_indexes": word_indexes,
                "flat_indexes": flat_indexes,
                "model_weights": model_weights,
            })

        inputs = self._encode(queries, documents)
        sessions = getattr(self, "sessions", [self.session])
        logits_by_model = [
            np.asarray(session.run(["logits"], inputs)[0]).reshape(-1)
            for session in sessions
        ]
        if any(
            logits.shape != logits_by_model[0].shape
            for logits in logits_by_model[1:]
        ) or len(logits_by_model[0]) != len(queries):
            raise RuntimeError("batch model outputs did not match flattened candidates")

        results = []
        batch_explanations = []
        for data in segment_data:
            evidence_scores: dict[str, float] = {}
            scored: list[tuple[float, int, str]] = []
            candidate_text_by_id: dict[str, str] = {}
            candidate_texts = [str(item["text"]) for item in data["candidates"]]
            model_weights = np.asarray(data["model_weights"], dtype=np.float32)
            for original_index, candidate in enumerate(data["candidates"]):
                candidate_id = str(candidate["id"])
                word = str(candidate["text"])
                flat_index = data["flat_indexes"][word]
                model_scores = np.asarray(
                    [logits[flat_index] for logits in logits_by_model],
                    dtype=np.float32,
                )
                raw_score = float(np.dot(model_scores, model_weights))
                lexical_penalty = (
                    self.lexicon.compute_lexical_penalty(word, data["reading"])
                    if getattr(self, "lexicon", None) and data["reading"]
                    else 0.0
                )
                style_bonus = orthographic_style_bonus(word, candidate_texts)
                context_bonus = contextual_candidate_bonus(
                    data["prefix"], data["suffix"], word
                )
                reading_penalty = reading_identity_penalty(
                    word, data["reading"], candidate_texts
                )
                evidence_score = (
                    raw_score
                    + style_bonus
                    + context_bonus
                    - lexical_penalty
                    - reading_penalty
                )
                evidence_scores[candidate_id] = float(evidence_score)
                candidate_text_by_id[candidate_id] = word
                final_score = evidence_score - _rank_prior_penalty(
                    original_index, getattr(self, "prior_w", 0.0)
                )
                scored.append((float(final_score), original_index, candidate_id))

            scored.sort(key=lambda item: item[0], reverse=True)
            neural_top_id = scored[0][2]
            neural_confidence = _neural_top_probability(
                evidence_scores, neural_top_id
            )
            if getattr(self, "safety_gate", True):
                scored = _preserve_mozc_top_if_uncertain(
                    scored,
                    data["prefix"],
                    data["suffix"],
                    evidence_scores,
                    candidate_text_by_id,
                    data["reading"],
                    getattr(self, "ensemble_gate_thresholds", None),
                )
            winner_id = scored[0][2]
            results.append({
                "id": data["id"],
                "winner_id": winner_id,
                "confidence": float(neural_confidence),
            })
            batch_explanations.append({
                "id": data["id"],
                "winner_id": winner_id,
                "neural_top_id": neural_top_id,
                "confidence": float(neural_confidence),
            })

        self.last_batch_latency_ms = round(
            (time.perf_counter() - started) * 1000.0, 2
        )
        self.last_batch_explanation = {"segments": batch_explanations}
        return {"request_id": request["request_id"], "segments": results}
