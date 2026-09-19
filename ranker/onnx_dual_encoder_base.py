"""Ultra-low latency Dual-Encoder ONNX Runtime backend for Yamatana AI IME.

Operates purely with onnxruntime and tokenizers (no PyTorch/Transformers required),
achieving <10ms inference latency with safe boundaries, grammatical particle rules,
and resident in-memory candidate vector cache.
"""

from __future__ import annotations

import logging
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_settings import load_settings, normalize_settings, product_data_dir
from ranker.scoring import contextual_candidate_bonus, reading_identity_penalty
from ranker.persistent_embedding_store import PersistentEmbeddingStore

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
        candidate_store_path: Optional[str | Path] = None,
        candidate_warmup_limit: Optional[int] = None,
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

        # In-memory candidate embedding cache (word -> 384-dim numpy array).
        # The persistent store below is float16 on disk; scoring remains
        # float32 so the dot-product path does not lose precision.
        self.candidate_cache: dict[str, np.ndarray] = {}
        # Context vectors are keyed by their exact safe context string.  A
        # changed context therefore cannot reuse a stale vector, while a
        # batch-size change does not evict an identical prefetched context.
        self.context_cache: dict[str, np.ndarray] = {}
        self.context_cache_signature: tuple[str, ...] = ()
        self._cache_lock = threading.RLock()
        self._model_lock = threading.Lock()
        model_stat = Path(self.model_path).stat()
        model_key = (
            f"{self.model_path}|{model_stat.st_size}|{model_stat.st_mtime_ns}"
        )
        store_override = os.environ.get("YAMATANA_CANDIDATE_STORE")
        store_path = Path(
            candidate_store_path
            or store_override
            or (product_data_dir() / "candidate_embeddings.sqlite3")
        )
        self.candidate_store = PersistentEmbeddingStore(store_path, model_key)
        configured_warmup = os.environ.get("YAMATANA_CANDIDATE_WARMUP_LIMIT")
        if candidate_warmup_limit is not None:
            self.candidate_warmup_limit = max(0, int(candidate_warmup_limit))
        elif configured_warmup is not None:
            try:
                self.candidate_warmup_limit = max(0, int(configured_warmup))
            except ValueError:
                self.candidate_warmup_limit = 100_000
        else:
            # Standalone/unit-test construction stays lazy.  The resident
            # named-pipe process opts into the 100k startup warmup explicitly
            # below, so one-shot callers never launch a bulk ONNX worker.
            self.candidate_warmup_limit = 0

        self._prefetch_state_lock = threading.Lock()
        self._prefetch_condition = threading.Condition(self._prefetch_state_lock)
        self._pending_context_prefetch: Optional[Dict[str, Any]] = None
        self._context_prefetch_worker: Optional[threading.Thread] = None
        self._pending_candidate_prefetch: Optional[Dict[str, Any]] = None
        self._candidate_prefetch_worker: Optional[threading.Thread] = None
        self._context_prefetch_signature: tuple[str, ...] = ()

        # This is deliberately best-effort and daemonized.  It fills the
        # persistent dictionary from the packaged homophone vocabulary while
        # the tray is idle; live candidate/context prefetch takes priority.
        self._candidate_warmup_worker: Optional[threading.Thread] = None
        if self.candidate_warmup_limit > 0:
            self._candidate_warmup_worker = threading.Thread(
                target=self._warm_candidate_dictionary,
                name="yamatana-candidate-warmup",
                daemon=True,
            )
            self._candidate_warmup_worker.start()

    def encode_texts(self, texts: list[str], max_length: int = 48) -> np.ndarray:
        """Encode a batch of texts into normalized embedding vectors (N x 384)."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Tokenizer truncation is mutable, so serialize model calls while
        # allowing the pipe thread and the latest-prefetch worker to coexist.
        with self._model_lock:
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
        """Pre-compute and cache candidate embeddings in memory and on disk."""
        with self._cache_lock:
            missing = [w for w in set(words) if w and w not in self.candidate_cache]
        if not missing:
            return
        persisted = self.candidate_store.get_many(missing)
        if persisted:
            with self._cache_lock:
                self.candidate_cache.update(persisted)
            missing = [word for word in missing if word not in persisted]
        if not missing:
            return
        embeddings = self.encode_texts(missing, max_length=16)
        with self._cache_lock:
            for w, emb in zip(missing, embeddings):
                self.candidate_cache[w] = emb
        self.candidate_store.put_many(missing, embeddings)

    def get_candidate_embedding(self, word: str) -> np.ndarray:
        """Get candidate embedding with transparent in-memory caching."""
        with self._cache_lock:
            cached = self.candidate_cache.get(word)
        if cached is not None:
            return cached
        persisted = self.candidate_store.get_many([word]).get(word)
        if persisted is not None:
            with self._cache_lock:
                self.candidate_cache[word] = persisted
            return persisted
        emb = self.encode_texts([word], max_length=16)[0]
        with self._cache_lock:
            self.candidate_cache[word] = emb
        self.candidate_store.put_many([word], emb.reshape(1, -1))
        return emb

    def _get_context_vectors(self, context_queries: Sequence[str]) -> np.ndarray:
        """Return cached context vectors, encoding only cache misses."""
        queries = tuple(str(query) for query in context_queries)
        with self._cache_lock:
            self.context_cache_signature = queries
            missing = [query for query in dict.fromkeys(queries)
                       if query not in self.context_cache]
        if missing:
            embeddings = self.encode_texts(missing, max_length=self.context_chars)
            with self._cache_lock:
                for query, embedding in zip(missing, embeddings):
                    self.context_cache[query] = embedding
        if not queries:
            return np.empty((0, 384), dtype=np.float32)
        with self._cache_lock:
            return np.stack([self.context_cache[query] for query in queries], axis=0)

    def _get_cached_context_vectors(
        self, context_queries: Sequence[str]
    ) -> Optional[np.ndarray]:
        """Read context vectors without encoding or mutating the cache.

        This is the Space/explicit-conversion path.  It only reads the cache;
        the separate wait helper may wait for an already-running prefetch, but
        it never starts a model forward pass on the named-pipe request thread.
        """
        queries = tuple(str(query) for query in context_queries)
        if not queries:
            return np.empty((0, 384), dtype=np.float32)
        with self._cache_lock:
            if any(query not in self.context_cache for query in queries):
                return None
            return np.stack([self.context_cache[query] for query in queries], axis=0)

    def _clear_context_cache(self) -> None:
        with self._cache_lock:
            self.context_cache.clear()
            self.context_cache_signature = ()

    def _wait_for_prefetch_cache(
        self,
        context_queries: Sequence[str],
        candidate_words: Sequence[str],
        *,
        timeout_seconds: float = 0.35,
    ) -> bool:
        """Wait for an already-running prefetch, without doing model work here."""
        contexts = tuple(str(query) for query in context_queries)
        words = tuple(str(word) for word in candidate_words if str(word))
        deadline = time.perf_counter() + max(0.0, float(timeout_seconds))

        while True:
            with self._cache_lock:
                ready = (
                    all(query in self.context_cache for query in contexts)
                    and all(word in self.candidate_cache for word in words)
                )
            if ready:
                return True

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False
            with self._prefetch_condition:
                context_needed = any(
                    query not in self.context_cache for query in contexts
                )
                candidate_needed = any(
                    word not in self.candidate_cache for word in words
                )
                context_active = (
                    self._context_prefetch_worker is not None
                    and self._context_prefetch_worker.is_alive()
                ) or self._pending_context_prefetch is not None
                candidate_active = (
                    self._candidate_prefetch_worker is not None
                    and self._candidate_prefetch_worker.is_alive()
                ) or self._pending_candidate_prefetch is not None
                if ((not context_needed or not context_active) and
                        (not candidate_needed or not candidate_active)):
                    return False
                self._prefetch_condition.wait(timeout=remaining)

    def _prefetch_active(self) -> bool:
        """Return whether a background prefetch is queued or running."""
        with self._prefetch_condition:
            return (
                self._pending_context_prefetch is not None
                or self._pending_candidate_prefetch is not None
                or (self._context_prefetch_worker is not None
                    and self._context_prefetch_worker.is_alive())
                or (self._candidate_prefetch_worker is not None
                    and self._candidate_prefetch_worker.is_alive())
            )

    def _candidate_vocabulary_path(self) -> Optional[Path]:
        override = os.environ.get("YAMATANA_CANDIDATE_VOCABULARY")
        if override:
            path = Path(override)
            if path.exists():
                return path
        return _resolve_first((
            "data/candidate_vocabulary.txt",
            "data/massive_homophone_database.json",
        ))

    def _iter_candidate_vocabulary(self):
        """Yield unique surfaces from the packaged candidate vocabulary."""
        path = self._candidate_vocabulary_path()
        if path is None:
            return
        seen: set[str] = set()
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    database = json.load(handle)
                for entry in database.values():
                    for candidate in entry.get("candidates", []):
                        word = str(candidate).strip()
                        if word and word not in seen:
                            seen.add(word)
                            yield word
                return
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    word = line.strip()
                    if word and not word.startswith("#") and word not in seen:
                        seen.add(word)
                        yield word
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
            LOG.warning("could not read candidate vocabulary %s: %s", path, exc)

    def _warm_candidate_dictionary(self) -> None:
        """Fill the resident/persistent candidate dictionary opportunistically."""
        # Let the pipe become ready before doing any bulk work.
        time.sleep(0.2)
        batch: list[str] = []
        seen_count = 0
        batch_size = 256
        try:
            for word in self._iter_candidate_vocabulary():
                if seen_count >= self.candidate_warmup_limit:
                    break
                batch.append(word)
                seen_count += 1
                if len(batch) < batch_size:
                    continue
                self._wait_for_live_prefetch()
                self.preload_candidates(batch)
                batch.clear()
            if batch:
                self._wait_for_live_prefetch()
                self.preload_candidates(batch)
            LOG.info(
                "candidate dictionary warmup complete: %s words, persistent=%s, stored=%s",
                seen_count, self.candidate_store.enabled, self.candidate_store.count(),
            )
        except Exception:
            LOG.exception("candidate dictionary warmup failed")

    def _wait_for_live_prefetch(self) -> None:
        """Yield bulk work while live context/candidate prefetch is active."""
        while True:
            with self._prefetch_condition:
                active = (
                    self._pending_context_prefetch is not None
                    or self._pending_candidate_prefetch is not None
                    or (self._context_prefetch_worker is not None
                        and self._context_prefetch_worker.is_alive())
                    or (self._candidate_prefetch_worker is not None
                        and self._candidate_prefetch_worker.is_alive())
                )
            if not active:
                return
            time.sleep(0.02)

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
        *,
        allow_encode: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        words = [str(c.get("text", c.get("word", ""))) for c in candidates]
        if allow_encode:
            self.preload_candidates(words)
        with self._cache_lock:
            missing = (words if not allow_encode
                       else [word for word in words if word])
            if any(word not in self.candidate_cache for word in missing):
                return None

        cand_vectors = np.stack([self.get_candidate_embedding(w) for w in words], axis=0)  # (K, 384)
        cos_sims = np.dot(cand_vectors, ctx_vector)  # (K,)

        scored_candidates = []
        for idx, cand in enumerate(candidates):
            word = words[idx]
            original_rank = int(cand.get("rank", idx + 1))
            raw_sim = float(cos_sims[idx])

            gram_bonus = self.compute_grammatical_particle_bonus(prefix, word)
            context_bonus = (
                contextual_candidate_bonus(prefix, suffix, word)
                if self.enable_lexical_grounding
                else 0.0
            )
            mora_penalty = self.compute_length_and_mora_penalty(word, reading)
            reading_penalty = reading_identity_penalty(word, reading, words)
            rank_prior = max(0.0, 0.04 * (len(candidates) - original_rank) / max(1, len(candidates)))

            final_score = (
                raw_sim * 2.0
                + gram_bonus
                + context_bonus
                - mora_penalty
                - reading_penalty
                + rank_prior
            )

            scored_item = dict(cand)
            scored_item["score"] = round(raw_sim, 4)
            scored_item["final_score"] = round(final_score, 4)
            scored_item["cosine_sim"] = round(raw_sim, 4)
            scored_item["gram_bonus"] = round(gram_bonus, 4)
            scored_item["context_bonus"] = round(context_bonus, 4)
            scored_item["mora_penalty"] = round(mora_penalty, 4)
            scored_item["reading_penalty"] = round(reading_penalty, 4)
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
            self._clear_context_cache()
            clean_fallback = [
                {
                    "id": str(c["id"]),
                    "score": float(c.get("score", -int(c.get("rank", idx + 1)))),
                    "rank": int(c.get("rank", idx + 1)),
                }
                for idx, c in enumerate(candidates)
            ]
            return {
                "request_id": request_id,
                "candidates": clean_fallback,
            }

        # 2. Context Vector Encoding (or predictor-prefetched cache hit)
        ctx_vector = self._get_context_vectors([context_query])[0]

        # 3. Dot-Product Scoring
        scored = self._score_segment_candidates(local_prefix, suffix, reading, candidates, ctx_vector)

        clean_scored = [
            {
                "id": str(c["id"]),
                "score": float(c["score"]),
                "rank": int(c["rank"]),
            }
            for c in scored
        ]
        return {
            "request_id": request_id,
            "candidates": clean_scored,
        }

    def rank_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rank multiple segments simultaneously using ONE-SHOT batch context encoding and ALL-DOT-PRODUCT."""
        request_id = request.get("request_id", "batch")
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request_id, "segments": []}

        # 1. Collect safe context for every segment.  Keep a stable query key
        # for each segment so a batch-size change does not evict a still-valid
        # prefetched context vector.
        context_queries = []
        safe_prefixes = []
        for seg in segments:
            p = str(seg.get("preceding_text", ""))
            safe_p = select_safe_local_context(p, max_chars=self.context_chars)
            safe_prefixes.append(safe_p)
            context_queries.append(
                safe_p if content_signal_length(safe_p) >= 2 else "文脈"
            )

        # 2. ONE-SHOT Context Matrix Encoding for misses only.
        context_matrix = self._get_context_vectors(context_queries)

        # 3. Batch Dot-Product Scoring for ALL segments
        output_segments = []

        for s_idx, seg in enumerate(segments):
            seg_id = seg.get("id", f"s{s_idx}")
            reading = str(seg.get("read", ""))
            candidates = seg.get("candidates", [])
            ctx_vec = context_matrix[s_idx]
            local_prefix = safe_prefixes[s_idx]
            suffix = str(seg.get("following_text", ""))

            if not local_prefix or not candidates or content_signal_length(local_prefix) < 2:
                winner_id = str(candidates[0].get("id", "c0")) if candidates else ""
                output_segments.append({
                    "id": seg_id,
                    "winner_id": winner_id,
                    "confidence": 0.5,
                })
                continue

            scored = self._score_segment_candidates(local_prefix, suffix, reading, candidates, ctx_vec)

            # Softmax confidence scaled by temperature factor (x4) so sharp margin yields >0.65
            scores = [float(item["final_score"]) * 4.0 for item in scored]
            max_s = max(scores)
            exp_s = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_s)
            confidence = (exp_s[0] / sum_exp) if sum_exp > 0 else 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            output_segments.append({
                "id": seg_id,
                "winner_id": str(scored[0]["id"]),
                "confidence": round(confidence, 4),
            })

        return {
            "request_id": request_id,
            "segments": output_segments,
        }

    @staticmethod
    def _prefetch_response(request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "request_id": request.get("request_id", "prefetch"),
            "segments": [
                {
                    "id": str(seg["id"]),
                    "winner_id": str(seg["candidates"][0]["id"]),
                    "confidence": 0.0,
                }
                for seg in request.get("segments", [])
            ],
        }

    def prefetch_context_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Warm only context vectors for the current conversion snapshot."""
        request_id = request.get("request_id", "prefetch")
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request_id, "segments": []}

        context_queries = []
        for seg in segments:
            safe_prefix = select_safe_local_context(
                str(seg.get("preceding_text", "")), max_chars=self.context_chars
            )
            context_queries.append(
                safe_prefix if content_signal_length(safe_prefix) >= 2 else "文脈"
            )
        signature = tuple(context_queries)
        with self._cache_lock:
            if signature != self._context_prefetch_signature:
                # Context vectors are tied to the current document context.
                # Never retain an older context generation after it changes.
                self.context_cache.clear()
                self.context_cache_signature = ()
                self._context_prefetch_signature = signature
        self._get_context_vectors(context_queries)
        return self._prefetch_response(request)

    def prefetch_candidate_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Warm only candidate vectors; context encoding is never triggered."""
        segments = request.get("segments", [])
        if not segments:
            return {"request_id": request.get("request_id", "prefetch"),
                    "segments": []}
        candidate_words = [
            str(candidate.get("text", candidate.get("word", "")))
            for seg in segments
            for candidate in seg.get("candidates", [])
            if str(candidate.get("text", candidate.get("word", "")))
        ]
        self.preload_candidates(candidate_words)
        return self._prefetch_response(request)

    def prefetch_batch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible combined prefetch for older Mozc binaries."""
        self.prefetch_context_batch(request)
        return self.prefetch_candidate_batch(request)

    def prefetch_batch_async(self, request: Dict[str, Any]) -> None:
        """Queue both sides for compatibility with older clients."""
        self.prefetch_context_batch_async(request)
        self.prefetch_candidate_batch_async(request)

    def prefetch_context_batch_async(self, request: Dict[str, Any]) -> None:
        """Queue the newest context snapshot without blocking the pipe."""
        self._queue_prefetch(request, context=True)

    def prefetch_candidate_batch_async(self, request: Dict[str, Any]) -> None:
        """Queue the newest candidate range without blocking the pipe."""
        self._queue_prefetch(request, context=False)

    def _queue_prefetch(self, request: Dict[str, Any], *, context: bool) -> None:
        with self._prefetch_condition:
            if context:
                self._pending_context_prefetch = request
                worker = self._context_prefetch_worker
            else:
                self._pending_candidate_prefetch = request
                worker = self._candidate_prefetch_worker
            if worker is not None and worker.is_alive():
                return
            worker = threading.Thread(
                target=self._run_pending_prefetch,
                args=(context,),
                name="yamatana-context-prefetch" if context
                else "yamatana-candidate-prefetch",
                daemon=True,
            )
            if context:
                self._context_prefetch_worker = worker
            else:
                self._candidate_prefetch_worker = worker
            worker.start()

    def _run_pending_prefetch(self, context: bool) -> None:
        while True:
            with self._prefetch_state_lock:
                if context:
                    request = self._pending_context_prefetch
                    self._pending_context_prefetch = None
                else:
                    request = self._pending_candidate_prefetch
                    self._pending_candidate_prefetch = None
            if request is None:
                with self._prefetch_condition:
                    if context:
                        self._context_prefetch_worker = None
                    else:
                        self._candidate_prefetch_worker = None
                    self._prefetch_condition.notify_all()
                return
            try:
                if context:
                    self.prefetch_context_batch(request)
                else:
                    self.prefetch_candidate_batch(request)
            except Exception:
                LOG.exception(
                    "%s prefetch failed",
                    "context" if context else "candidate",
                )
            finally:
                with self._prefetch_condition:
                    self._prefetch_condition.notify_all()
