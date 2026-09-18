"""Ultra-low latency Dual-Encoder ONNX Runtime backend for Yamatana AI IME.

Operates purely with onnxruntime and tokenizers (no PyTorch/Transformers required),
achieving <10ms inference latency with safe boundaries, grammatical particle rules,
and resident in-memory candidate vector cache.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_settings import load_settings, normalize_settings

LOG = logging.getLogger("yamatana_ai_ime.onnx_dual_encoder")

_CJK_RE = re.compile(r"[一-龯々〆ヵヶ]")
_HIRAGANA_RE = re.compile(r"^[ぁ-ゖー・]+$")
_CONTEXT_NOISE_CHARS = set("これそれあれこのそのあの \t\r\n、。,.!?！？「」『』（）()［］[]【】{}・:：;；")


def estimate_mora_length(reading: str) -> int:
    """Approximate the mora count of Japanese reading (hiragana)."""
    if not reading:
        return 0
    small_kana = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")
    return sum(1 for char in reading if char not in small_kana)


def select_safe_local_context(preceding_text: str, max_chars: int = 36) -> str:
    """Safely select local context window without prematurely severing at recent commas.
    
    Safety Policy:
    - Full sentence terminations (。, ！？, newlines) are absolute boundaries.
    - Commas (、) are weak boundaries: NEVER cut at a comma within the last 18-20 characters!
    - Retain up to max_chars (default 36 chars) of recent context.
    """
    if not preceding_text or max_chars <= 0:
        return ""
    
    text = str(preceding_text).strip()
    hard_terminators = "\r\n。！？!?"
    last_hard = max((text.rfind(c) for c in hard_terminators), default=-1)
    if last_hard >= 0:
        text = text[last_hard + 1:].strip()
        
    if len(text) > 20:
        comma_positions = [i for i, c in enumerate(text) if c in "、,"]
        safe_commas = [pos for pos in comma_positions if (len(text) - pos) >= 18]
        if safe_commas:
            text = text[safe_commas[-1] + 1:].strip()

    if len(text) > max_chars:
        text = text[-max_chars:].strip()

    return text


def content_signal_length(text: str) -> int:
    """Calculate meaningful semantic signal length excluding noise and demonstratives."""
    return sum(1 for c in str(text) if c not in _CONTEXT_NOISE_CHARS)


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


class OnnxDualEncoderIMEReranker:
    """Ultra-low latency IME reranker based on Dual-Encoder ONNX, dot-product vector search,
    safe boundary selection, and multi-segment batch dot-product."""

    def __init__(
        self,
        settings: Optional[Mapping[str, Any]] = None,
        settings_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        context_chars: int = 36,
        execution_providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.settings = normalize_settings(settings) if settings is not None else load_settings(settings_path)
        self.context_enabled = bool(self.settings.get("context_enabled", True))
        self.context_chars = int(self.settings.get("context_chars", context_chars))
        self.document_domain = str(self.settings.get("document_domain", "general"))
        self.enable_lexical_grounding = bool(self.settings.get("lexical_grounding", True))

        available = set(ort.get_available_providers())
        requested = str(self.settings.get("compute_mode", "auto"))
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

        if model_path is not None:
            resolved_model = Path(model_path)
        elif use_gpu:
            resolved_model = _resolve_first((
                "build/onnx-model-70m-dual-encoder/dual-encoder-70m-fp16.onnx",
                "models/onnx/dual-encoder-70m-fp16.onnx",
                "build/onnx-model-70m-dual-encoder/dual-encoder-70m-fp32.onnx",
                "models/onnx/dual-encoder-70m-fp32.onnx",
            ))
        else:
            resolved_model = _resolve_first((
                "build/onnx-model-70m-dual-encoder/dual-encoder-70m-int8.onnx",
                "models/onnx/dual-encoder-70m-int8.onnx",
                "build/onnx-model-70m-dual-encoder/dual-encoder-70m-fp16.onnx",
                "models/onnx/dual-encoder-70m-fp16.onnx",
            ))

        if resolved_model is None or not resolved_model.exists():
            raise FileNotFoundError(f"Dual-Encoder ONNX model not found (searched roots: {_runtime_roots()})")

        self.model_path = str(resolved_model.resolve())
        tokenizer_path = _resolve_first((
            "build/onnx-model-70m-dual-encoder/tokenizer.json",
            "models/onnx/tokenizer.json",
            "build/dual-encoder-70m/tokenizer.json",
            "build/onnx-model-70m/tokenizer.json",
        ))
        if tokenizer_path is None or not tokenizer_path.exists():
            raise FileNotFoundError("Dual-Encoder tokenizer.json not found")

        LOG.info("Loading Dual-Encoder tokenizer from %s...", tokenizer_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=64)
        self.tokenizer.enable_padding(pad_id=3, pad_token="<pad>")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = min(8, max(2, os.cpu_count() or 2))
        options.inter_op_num_threads = 1

        if provider_override:
            providers = provider_override
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
        elif use_gpu:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [gpu_provider, "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        LOG.info("Initializing Dual-Encoder ONNX session from %s with providers %s...", self.model_path, providers)
        self.session = ort.InferenceSession(self.model_path, options, providers=providers)
        self.device = "gpu" if use_gpu else "cpu"

        # In-memory candidate embedding cache (word -> 384-dim numpy array)
        self.candidate_cache: dict[str, np.ndarray] = {}

    def encode_texts(self, texts: list[str], max_length: int = 48) -> np.ndarray:
        """Encode a batch of texts into normalized embedding vectors (N x 384)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        self.tokenizer.enable_truncation(max_length=max_length)
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        outputs = self.session.run(["embeddings"], ort_inputs)[0]
        return outputs.astype(np.float32)

    def preload_candidates(self, words: Sequence[str]) -> None:
        """Pre-compute and cache candidate embeddings in resident memory."""
        missing = [w for w in set(words) if w not in self.candidate_cache]
        if not missing:
            return
        embeddings = self.encode_texts(missing, max_length=16)
        for w, emb in zip(missing, embeddings):
            self.candidate_cache[w] = emb

    def get_candidate_embedding(self, word: str) -> np.ndarray:
        """Get candidate embedding with transparent in-memory caching."""
        if word in self.candidate_cache:
            return self.candidate_cache[word]
        emb = self.encode_texts([word], max_length=16)[0]
        self.candidate_cache[word] = emb
        return emb

    def compute_length_and_mora_penalty(self, word: str, reading: str) -> float:
        """Penalize candidate words whose character length departs from expected reading."""
        if not reading or not word:
            return 0.0
        mora = estimate_mora_length(reading)
        kanji_count = sum(1 for c in word if _CJK_RE.match(c))
        word_len = len(word)

        penalty = 0.0
        expected_kanji = max(1, int(math.ceil(mora / 2.0)))
        if kanji_count > expected_kanji + 1:
            penalty += (kanji_count - expected_kanji) * 0.25

        if word_len > mora + 2:
            penalty += (word_len - mora - 1) * 0.20

        return penalty

    def compute_grammatical_particle_bonus(self, prefix: str, word: str) -> float:
        """Apply linguistic rules for case particles (格助詞) and delicate homophones."""
        bonus = 0.0
        p = prefix.strip()
        if not p or not word:
            return 0.0

        # Rule 1: Accusative [〜を] takes Transitive verbs (他動詞)
        if p.endswith("を") or p.endswith("を、"):
            if word in {"空ける", "揚げる", "治す", "収める", "決裁", "精算", "清算", "校正", "制作", "映す", "侵す"}:
                bonus += 0.20
            if word in {"明ける", "上がる", "直る", "治まる"}:
                bonus -= 0.35

        # Rule 2: Nominative [〜が] takes Intransitive verbs (自動詞)
        if p.endswith("が") or p.endswith("が、"):
            if word in {"明ける", "上がる", "直る", "治まる", "降る", "鳴る", "開く", "閉まる"}:
                bonus += 0.20
            if word in {"空ける", "揚げる", "治す", "収める"}:
                bonus -= 0.35

        # Rule 3: Locative/Directional [〜に]
        if p.endswith("に") or p.endswith("に、"):
            if word in {"移行", "参加", "行く", "着く", "入る", "住む", "写す"}:
                bonus += 0.15
            if word in {"以降"}:
                bonus -= 0.30

        # Rule 4: Deliberate homophone contextual pairs
        if "クラウド" in p or "サーバー" in p or "システム" in p or "環境" in p:
            if word == "移行":
                bonus += 0.30
            elif word == "以降":
                bonus -= 0.25

        if "稟議" in p or "役員" in p or "上長" in p or "申請" in p or "承認" in p:
            if word == "決裁":
                bonus += 0.35
            elif word == "決済":
                bonus -= 0.25

        if "病気" in p or "風邪" in p or "怪我" in p or "医師" in p or "薬" in p:
            if word == "治す":
                bonus += 0.35
            elif word == "直す":
                bonus -= 0.25

        if "顔" in p or "目" in p or "口" in p or "耳" in p:
            if word == "鼻":
                bonus += 0.35
            elif word == "花":
                bonus -= 0.25

        if "法律" in p or "制度" in p or "改正" in p or "条例" in p:
            if word == "施行":
                bonus += 0.35
            elif word == "思考":
                bonus -= 0.25

        if "天気" in p or "雨" in p or "雪" in p or "空" in p or "外" in p:
            if word == "降る":
                bonus += 0.35
            elif word == "フル":
                bonus -= 0.30

        return bonus

    def _score_segment_candidates(
        self,
        prefix: str,
        suffix: str,
        reading: str,
        candidates: List[Dict[str, Any]],
        ctx_vector: np.ndarray,
    ) -> List[Dict[str, Any]]:
        words = [str(c.get("text", c.get("word", ""))) for c in candidates]
        self.preload_candidates(words)

        cand_vectors = np.stack([self.get_candidate_embedding(w) for w in words], axis=0)  # (K, 384)
        cos_sims = np.dot(cand_vectors, ctx_vector)  # (K,)

        scored_candidates = []
        for idx, cand in enumerate(candidates):
            word = words[idx]
            original_rank = int(cand.get("rank", idx + 1))
            raw_sim = float(cos_sims[idx])

            gram_bonus = self.compute_grammatical_particle_bonus(prefix, word)
            mora_penalty = self.compute_length_and_mora_penalty(word, reading)
            rank_prior = max(0.0, 0.04 * (len(candidates) - original_rank) / max(1, len(candidates)))

            final_score = raw_sim * 2.0 + gram_bonus - mora_penalty + rank_prior

            scored_item = dict(cand)
            scored_item["score"] = round(raw_sim, 4)
            scored_item["final_score"] = round(final_score, 4)
            scored_item["cosine_sim"] = round(raw_sim, 4)
            scored_item["gram_bonus"] = round(gram_bonus, 4)
            scored_item["mora_penalty"] = round(mora_penalty, 4)
            scored_candidates.append(scored_item)

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        for new_rank, item in enumerate(scored_candidates, start=1):
            item["rank"] = new_rank
            item["score"] = float(item["final_score"])
        return scored_candidates

    def rank(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rank a single conversion segment using safe local context and dot-product."""
        started = time.perf_counter()
        request_id = request.get("request_id", 0)
        prefix = str(request.get("preceding_text", request.get("prefix", "")))
        suffix = str(request.get("following_text", request.get("suffix", "")))
        reading = str(request.get("read", request.get("reading", "")))
        candidates = request.get("candidates", [])

        if not candidates:
            return {"request_id": request_id, "candidates": []}

        # 1. Safe Context Selection
        local_prefix = select_safe_local_context(prefix, max_chars=self.context_chars)
        context_query = local_prefix.strip()

        # Zero or ambiguous context safety gate
        if not context_query or content_signal_length(context_query) < 2:
            first_text = str(candidates[0].get("text", candidates[0].get("word", "")))
            first_id = str(candidates[0].get("id", "c1"))
            fallback_cands = [
                {
                    "id": str(c["id"]),
                    "text": c.get("text", c.get("word", "")),
                    "score": float(c.get("score", -int(c.get("rank", idx + 1)))),
                    "rank": int(c.get("rank", idx + 1)),
                }
                for idx, c in enumerate(candidates)
            ]
            return {
                "request_id": request_id,
                "candidates": fallback_cands,
                "winner_id": first_id,
                "winner_text": first_text,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "breakdown": {"context_encoding_ms": 0.0, "dot_product_ms": 0.0},
            }

        # 2. Context Vector Encoding
        t_ctx_start = time.perf_counter()
        ctx_vector = self.encode_texts([context_query], max_length=self.context_chars)[0]
        ctx_latency_ms = (time.perf_counter() - t_ctx_start) * 1000.0

        # 3. Dot-Product Scoring
        t_dot_start = time.perf_counter()
        scored = self._score_segment_candidates(local_prefix, suffix, reading, candidates, ctx_vector)
        dot_latency_ms = (time.perf_counter() - t_dot_start) * 1000.0
        total_latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "request_id": request_id,
            "candidates": scored,
            "winner_id": scored[0]["id"],
            "winner_text": scored[0].get("text", scored[0].get("word", "")),
            "latency_ms": round(total_latency_ms, 3),
            "breakdown": {
                "context_encoding_ms": round(ctx_latency_ms, 3),
                "dot_product_ms": round(dot_latency_ms, 3),
            },
        }

    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rank multiple segments simultaneously using ONE-SHOT batch context encoding and ALL-DOT-PRODUCT."""
        started = time.perf_counter()
        request_id = request.get("request_id", "batch")
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request_id, "segments": []}

        # 1. Collect safe context for every segment
        context_queries = []
        safe_prefixes = []
        for seg in segments:
            p = str(seg.get("preceding_text", ""))
            safe_p = select_safe_local_context(p, max_chars=self.context_chars)
            safe_prefixes.append(safe_p)
            context_queries.append(safe_p if safe_p else "文脈")

        # 2. ONE-SHOT Context Matrix Encoding: (S x 384) in a single fast forward pass
        t_ctx_start = time.perf_counter()
        context_matrix = self.encode_texts(context_queries, max_length=self.context_chars)
        ctx_latency_ms = (time.perf_counter() - t_ctx_start) * 1000.0

        # 3. Batch Dot-Product Scoring for ALL segments
        t_dot_start = time.perf_counter()
        output_segments = []

        for s_idx, seg in enumerate(segments):
            seg_id = seg.get("id", f"s{s_idx}")
            reading = str(seg.get("read", ""))
            candidates = seg.get("candidates", [])
            ctx_vec = context_matrix[s_idx]
            local_prefix = safe_prefixes[s_idx]
            suffix = str(seg.get("following_text", ""))

            if not local_prefix or not candidates or content_signal_length(local_prefix) < 2:
                first_text = str(candidates[0].get("text", candidates[0].get("word", ""))) if candidates else ""
                output_segments.append({
                    "id": seg_id,
                    "winner_id": candidates[0].get("id", "c1") if candidates else "",
                    "winner_text": first_text,
                    "candidates": candidates,
                })
                continue

            scored = self._score_segment_candidates(local_prefix, suffix, reading, candidates, ctx_vec)
            output_segments.append({
                "id": seg_id,
                "winner_id": scored[0]["id"],
                "winner_text": scored[0].get("text", scored[0].get("word", "")),
                "confidence": round(float(scored[0]["final_score"]), 4),
                "candidates": scored,
            })

        dot_latency_ms = (time.perf_counter() - t_dot_start) * 1000.0
        total_latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "request_id": request_id,
            "segments": output_segments,
            "latency_ms": round(total_latency_ms, 3),
            "breakdown": {
                "segment_count": len(segments),
                "batch_context_encoding_ms": round(ctx_latency_ms, 3),
                "all_dot_product_ms": round(dot_latency_ms, 3),
            },
        }
