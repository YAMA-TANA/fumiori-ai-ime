"""Small persistent store for candidate-side dual-encoder vectors.

The IME process is intentionally resident, but the candidate encoder should
not have to rediscover the same word after every restart.  SQLite gives us an
atomic, crash-tolerant dictionary without adding a third-party dependency.
Vectors are stored as float16 blobs and converted to float32 only when they
enter the in-memory scoring cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable, Mapping, Optional

import numpy as np


LOG = logging.getLogger("yamatana_ai_ime.embedding_store")


class PersistentEmbeddingStore:
    """Model-versioned word -> embedding dictionary backed by SQLite."""

    def __init__(
        self,
        path: str | Path,
        model_key: str,
        *,
        dimension: int = 384,
    ) -> None:
        self.path = Path(path)
        self.model_key = str(model_key)
        self.dimension = int(dimension)
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(self.path), timeout=30.0, check_same_thread=False
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_embeddings (
                    model_key TEXT NOT NULL,
                    word TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (model_key, word)
                )
                """
            )
            # A model update invalidates every old vector.  Removing old
            # generations keeps the user's persistent dictionary bounded.
            self._connection.execute(
                "DELETE FROM candidate_embeddings WHERE model_key <> ?",
                (self.model_key,),
            )
            self._connection.commit()
        except (OSError, sqlite3.Error) as exc:
            LOG.warning("candidate embedding store disabled: %s", exc)
            self._connection = None

    @property
    def enabled(self) -> bool:
        return self._connection is not None

    def missing_words(self, words: Iterable[str]) -> list[str]:
        """Find vectors absent from this model generation without loading blobs."""
        unique = list(dict.fromkeys(str(word) for word in words if str(word)))
        if not unique or self._connection is None:
            return unique
        present: set[str] = set()
        try:
            with self._lock:
                for offset in range(0, len(unique), 400):
                    chunk = unique[offset:offset + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = self._connection.execute(
                        f"SELECT word FROM candidate_embeddings WHERE model_key = ? "
                        f"AND dimension = ? AND length(embedding) = ? "
                        f"AND word IN ({placeholders})",
                        [self.model_key, self.dimension, self.dimension * 2, *chunk],
                    ).fetchall()
                    present.update(str(row[0]) for row in rows)
        except sqlite3.Error as exc:
            LOG.warning("could not inspect candidate embedding store: %s", exc)
            return unique
        return [word for word in unique if word not in present]

    def get_many(self, words: Iterable[str]) -> Mapping[str, np.ndarray]:
        unique = list(dict.fromkeys(str(word) for word in words if str(word)))
        if not unique or self._connection is None:
            return {}
        result: dict[str, np.ndarray] = {}
        try:
            with self._lock:
                # Stay below SQLite's parameter limit on older Windows builds.
                for offset in range(0, len(unique), 400):
                    chunk = unique[offset:offset + 400]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = self._connection.execute(
                        f"SELECT word, dimension, embedding "
                        f"FROM candidate_embeddings WHERE model_key = ? "
                        f"AND word IN ({placeholders})",
                        [self.model_key, *chunk],
                    ).fetchall()
                    for word, dimension, blob in rows:
                        if int(dimension) != self.dimension:
                            continue
                        vector = np.frombuffer(blob, dtype=np.float16)
                        if vector.size != self.dimension:
                            continue
                        result[str(word)] = vector.astype(np.float32, copy=True)
        except sqlite3.Error as exc:
            LOG.warning("could not read candidate embedding store: %s", exc)
        return result

    def put_many(self, words: Iterable[str], embeddings: np.ndarray) -> None:
        if self._connection is None:
            return
        unique = [str(word) for word in words if str(word)]
        values = np.asarray(embeddings)
        if not unique or values.ndim != 2 or values.shape[0] != len(unique):
            return
        if values.shape[1] != self.dimension:
            LOG.warning("ignoring candidate vectors with dimension %s", values.shape)
            return
        rows = [
            (
                self.model_key,
                word,
                self.dimension,
                np.asarray(vector, dtype=np.float16).tobytes(),
                time.time(),
            )
            for word, vector in zip(unique, values)
        ]
        try:
            with self._lock:
                self._connection.executemany(
                    """
                    INSERT OR REPLACE INTO candidate_embeddings
                    (model_key, word, dimension, embedding, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._connection.commit()
        except sqlite3.Error as exc:
            LOG.warning("could not write candidate embedding store: %s", exc)

    def count(self) -> int:
        if self._connection is None:
            return 0
        try:
            with self._lock:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM candidate_embeddings WHERE model_key = ?",
                    (self.model_key,),
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
