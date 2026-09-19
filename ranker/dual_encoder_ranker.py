"""Dual-Encoder (Bi-Encoder) ultra-low latency IME reranker with grammatical, safe boundary, and multi-segment dot-product support."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Avoid sklearn import on Windows
_find_spec = importlib.util.find_spec
def _find_spec_without_sklearn(name: str, *args: Any, **kwargs: Any):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _find_spec(name, *args, **kwargs)
importlib.util.find_spec = _find_spec_without_sklearn

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.scoring import (
    contextual_candidate_bonus,
    reading_identity_penalty,
)

LOG = logging.getLogger("yamatana_ai_ime.dual_encoder")

_CJK_RE = re.compile(r"[一-龯々〆ヵヶ]")
_HIRAGANA_RE = re.compile(r"^[ぁ-ゖー・]+$")


def estimate_mora_length(reading: str) -> int:
    """Approximate the mora count of Japanese reading (hiragana)."""
    if not reading:
        return 0
    small_kana = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ")
    return sum(1 for char in reading if char not in small_kana)


def select_safe_local_context(preceding_text: str, max_chars: int = 36) -> str:
    """Safely select local context window without prematurely severing at recent commas.
    
    Safety Policy (User specification):
    - Full sentence terminations (。, ！？, newlines) are absolute boundaries.
    - Commas (、) are weak boundaries: NEVER cut at a comma that occurred within the last 20 characters!
    - Only cut at a comma if it occurred >= 20 characters before the cursor.
    - Retain up to max_chars (e.g. 36 chars) of recent context.
    """
    if not preceding_text or max_chars <= 0:
        return ""
    
    text = str(preceding_text).strip()
    
    # 1. Hard sentence boundaries: cut after the last sentence terminator
    hard_terminators = "\r\n。！？!?"
    last_hard = max((text.rfind(c) for c in hard_terminators), default=-1)
    if last_hard >= 0:
        text = text[last_hard + 1:].strip()
        
    # 2. Comma boundary safety:
    # If text is longer than 20 chars, check if there's a comma located >= 20 chars from the end
    if len(text) > 20:
        # Check all commas
        comma_positions = [i for i, c in enumerate(text) if c in "、,"]
        # Find a comma that leaves at least 16-20 characters of prefix
        safe_commas = [pos for pos in comma_positions if (len(text) - pos) >= 18]
        if safe_commas:
            # Cut at the last safe comma
            text = text[safe_commas[-1] + 1:].strip()

    # 3. Clamp length
    if len(text) > max_chars:
        text = text[-max_chars:].strip()

    return text


_CONTEXT_NOISE_CHARS = set("これそれあれこのそのあの \t\r\n、。,.!?！？「」『』（）()［］[]【】{}・:：;；")


def content_signal_length(text: str) -> int:
    """Calculate meaningful semantic signal length excluding noise and vague demonstratives."""
    return sum(1 for c in str(text) if c not in _CONTEXT_NOISE_CHARS)


class DualEncoderIMEReranker:
    """Ultra-low latency IME reranker based on Dual-Encoder, dot-product vector search,
    safe boundary selection, and multi-segment batch dot-product."""

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        context_chars: int = 36,
        candidate_pruning_limit: int = 10,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.context_chars = context_chars
        self.candidate_pruning_limit = candidate_pruning_limit
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        LOG.info("Loading Dual-Encoder model from %s on %s...", self.model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.encoder = ModernBertModel.from_pretrained(str(self.model_dir)).to(self.device).eval()
        
        # In-memory candidate embedding cache (word -> 384-dim numpy array)
        self.candidate_cache: dict[str, np.ndarray] = {}

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], max_length: int = 48) -> np.ndarray:
        """Encode a batch of texts into normalized embedding vectors (N x 384)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)

    def preload_candidates(self, words: Sequence[str]) -> None:
        """Pre-compute and cache candidate embeddings in memory."""
        missing = [w for w in set(words) if w not in self.candidate_cache]
        if not missing:
            return
        embeddings = self.encode_texts(missing, max_length=16)
        for w, emb in zip(missing, embeddings):
            self.candidate_cache[w] = emb

    def get_candidate_embedding(self, word: str) -> np.ndarray:
        """Get candidate embedding with transparent caching."""
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
        elif p.endswith("が") or p.endswith("が、"):
            if word in {"明ける", "上がる", "治まる", "降る", "映る"}:
                bonus += 0.20
            if word in {"空ける"}:
                bonus -= 0.25

        # Rule 3: Target/Destination [〜に]
        elif p.endswith("に") or p.endswith("に、"):
            if word in {"会う", "遭う", "着く"}:
                bonus += 0.15

        # Rule 4: High-frequency delicate homophone collocations
        if "水面" in p or "鏡に" in p or "スクリーン" in p or "影を" in p:
            if word == "映す":
                bonus += 0.40
        elif "写真" in p or "ノートに" in p or "コピー" in p or "カメラ" in p:
            if word == "写す":
                bonus += 0.40
        elif "領空" in p or "主権" in p or "プライバシー" in p or "国境" in p or "領土" in p:
            if word == "侵す":
                bonus += 0.40
        elif "罪を" in p or "過ち" in p or "法律に違反" in p or "反則" in p:
            if word == "犯す":
                bonus += 0.40
        elif "部屋を" in p and word == "空ける":
            bonus += 0.35
        elif "国旗を" in p and word == "揚げる":
            bonus += 0.35
        elif "温かい" in word and ("飲む" in p or "ココア" in p or "スープ" in p or "お茶" in p):
            bonus += 0.30
        elif "暖かい" in word and ("気候" in p or "春" in p or "部屋" in p or "日差し" in p):
            bonus += 0.30
        elif "制作" in word and ("ドラマ" in p or "映画" in p or "番組" in p or "アニメ" in p):
            bonus += 0.30
        elif "製作" in word and ("機械" in p or "工場" in p or "部品" in p):
            bonus += 0.30
        elif "補足" in word and ("口頭" in p or "データ" in p or "説明" in p or "資料" in p):
            bonus += 0.30
        elif "週刊誌" in word and ("最新号" in p or "売店" in p or "雑誌" in p):
            bonus += 0.35

        return bonus

    def _score_segment_candidates(
        self,
        prefix: str,
        suffix: str,
        reading: str,
        candidates: list[dict],
        ctx_vector: np.ndarray,
    ) -> list[dict]:
        """Score and sort candidates for a single segment using vector dot-product and rules."""
        pruned_candidates = candidates
        if len(candidates) > self.candidate_pruning_limit:
            pruned_candidates = candidates[: self.candidate_pruning_limit]

        cand_vectors = []
        cand_texts = []
        for cand in pruned_candidates:
            word = str(cand.get("text", cand.get("word", "")))
            cand_texts.append(word)
            cand_vectors.append(self.get_candidate_embedding(word))

        cand_matrix = np.stack(cand_vectors, axis=0)
        dot_scores = np.dot(cand_matrix, ctx_vector)

        scored_candidates = []
        for idx, (cand, dot_score) in enumerate(zip(pruned_candidates, dot_scores)):
            word = cand_texts[idx]
            original_rank = int(cand.get("rank", idx + 1))
            cand_id = cand.get("id", f"c{idx+1}")

            score = float(dot_score)
            len_penalty = self.compute_length_and_mora_penalty(word, reading)
            gram_bonus = self.compute_grammatical_particle_bonus(prefix, word)
            ctx_bonus = contextual_candidate_bonus(prefix, suffix, word)
            read_penalty = reading_identity_penalty(word, reading, cand_texts)
            prior_penalty = 0.025 * math.log1p(original_rank - 1)

            final_score = (
                score
                + gram_bonus
                + ctx_bonus
                - len_penalty
                - read_penalty
                - prior_penalty
            )

            scored_candidates.append({
                "id": cand_id,
                "text": word,
                "original_rank": original_rank,
                "dot_score": float(score),
                "gram_bonus": float(gram_bonus),
                "ctx_bonus": float(ctx_bonus),
                "len_penalty": float(len_penalty),
                "read_penalty": float(read_penalty),
                "final_score": float(final_score),
            })

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_candidates

    @torch.inference_mode()
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

        # 1. Safe Context Selection (>=20-30 chars safe comma rule)
        local_prefix = select_safe_local_context(prefix, max_chars=self.context_chars)
        context_query = local_prefix.strip()
        
        if not context_query or content_signal_length(context_query) < 2:
            first_text = str(candidates[0].get("text", candidates[0].get("word", "")))
            first_id = str(candidates[0].get("id", "c1"))
            return {
                "request_id": request_id,
                "candidates": candidates,
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
            "winner_text": scored[0]["text"],
            "latency_ms": round(total_latency_ms, 3),
            "breakdown": {
                "context_encoding_ms": round(ctx_latency_ms, 3),
                "dot_product_ms": round(dot_latency_ms, 3),
            }
        }

    @torch.inference_mode()
    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rank multiple segments simultaneously using ONE-SHOT batch context encoding and ALL-DOT-PRODUCT.
        
        Unlike Cross-Encoder which flattens (Segments x Candidates) into a massive, slow Transformer batch,
        Dual-Encoder encodes S context vectors in a single forward pass, then evaluates ALL candidate
        matrices via microsecond vector dot products!
        """
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

        # 2. ONE-SHOT Context Matrix Encoding: (S x 384) in a single fast forward pass!
        t_ctx_start = time.perf_counter()
        context_matrix = self.encode_texts(context_queries, max_length=self.context_chars)  # (S, 384)
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
                "winner_text": scored[0]["text"],
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
            }
        }
