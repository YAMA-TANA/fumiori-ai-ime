"""Collect reproducible Japanese homophone data from public sources.

The collector intentionally keeps downloaded corpora outside git and emits a
small JSONL training shard.  It combines:

* JMdict (EDRDG): broad reading -> orthography groups and English glosses.
* Tatoeba Japanese sentences: naturally occurring contexts used to create
  positive/negative replacement pairs for the same reading.
* The curated 文化庁/異字同訓 groups already maintained in
  ``build_massive_homophone_db.py``.

The generated files are reproducible and contain source metadata.  JMdict and
Tatoeba terms must be retained when redistributing a trained artifact.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "homophone_sources"
JM_DICT_URL = "https://www.edrdg.org/pub/Nihongo/JMdict_e.gz"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2"


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading {url} -> {path}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "yamatana-ai-ime homophone collector"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def _is_japanese_word(word: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", word)) and not re.search(r"[<>(),]", word)


def parse_jmdict(path: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Return reading -> orthography -> glosses, retaining reading restrictions."""
    groups: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    with gzip.open(path, "rb") as stream:
        for event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag != "entry":
                continue
            kebs: List[str] = []
            for child in elem:
                if child.tag == "k_ele":
                    keb = child.findtext("keb")
                    if keb and _is_japanese_word(keb):
                        kebs.append(keb)
            rebs: List[Tuple[str, Set[str]]] = []
            for child in elem:
                if child.tag != "r_ele":
                    continue
                reb = child.findtext("reb")
                if not reb:
                    continue
                restrictions = {x.text for x in child.findall("re_restr") if x.text}
                rebs.append((_hiragana(reb), restrictions))
            glosses = {
                text.strip()
                for text in (g.text for g in elem.iter("gloss"))
                if text and text.strip()
            }
            for reading, restrictions in rebs:
                allowed = [k for k in kebs if not restrictions or k in restrictions]
                # Kana-only entries are useful for dictionary discovery but do
                # not form a kanji homophone training group by themselves.
                for keb in allowed:
                    groups[reading][keb].update(glosses)
            elem.clear()
    return groups


def write_groups(groups: Dict[str, Dict[str, Set[str]]], path: Path) -> int:
    records = []
    for reading, words in groups.items():
        candidates = sorted(words)
        if len(candidates) < 2 or len(candidates) > 40:
            continue
        records.append(
            {
                "reading": reading,
                "candidates": candidates,
                "glosses": {word: sorted(words[word])[:8] for word in candidates},
                "source": "JMdict/EDRDG",
            }
        )
    records.sort(key=lambda x: (x["reading"], x["candidates"]))
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def _load_group_records(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_tatoeba_pairs(
    sentences_path: Path,
    groups_path: Path,
    out_path: Path,
    max_pairs: int = 30000,
    max_per_word: int = 30,
) -> int:
    """Create natural context pairs by replacing a JMdict homophone.

    The original sentence is the positive candidate.  Every other spelling in
    the same JMdict reading group becomes a hard negative in the identical
    context.  We cap each spelling and the total shard to avoid over-weighting
    very common short words.
    """
    records = _load_group_records(groups_path)
    by_word: Dict[str, List[dict]] = {}
    for group in records:
        for word in group["candidates"]:
            by_word.setdefault(word, []).append(group)

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pairs: List[dict] = []
    with bz2.open(sentences_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if len(pairs) >= max_pairs:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3 or fields[1] != "jpn":
                continue
            sentence = fields[2].strip()
            if len(sentence) < 6 or len(sentence) > 120:
                continue
            for word, groups_for_word in by_word.items():
                if word not in sentence:
                    continue
                for group in groups_for_word:
                    reading = group["reading"]
                    key = (reading, word)
                    if counts[key] >= max_per_word:
                        continue
                    query = f"文脈「{sentence.replace(word, '____', 1)}」に最も適切な表記を選びなさい。"
                    positive = sentence
                    for negative_word in group["candidates"]:
                        if negative_word == word:
                            continue
                        pairs.append(
                            {
                                "reading": reading,
                                "query": query,
                                "positive": positive,
                                "negative": sentence.replace(word, negative_word, 1),
                                "source": "Tatoeba + JMdict",
                            }
                        )
                        if len(pairs) >= max_pairs:
                            break
                    counts[key] += 1
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pairs) + "\n", encoding="utf-8")
    return len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--max-per-word", type=int, default=30)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jmdict_path = _download(JM_DICT_URL, out_dir / "JMdict_e.gz")
    tatoeba_path = _download(TATOEBA_URL, out_dir / "jpn_sentences.tsv.bz2")

    groups = parse_jmdict(jmdict_path)
    groups_path = out_dir / "jmdict_homophone_groups.json"
    group_count = write_groups(groups, groups_path)
    pairs_path = out_dir / "tatoeba_homophone_pairs.jsonl"
    pair_count = build_tatoeba_pairs(
        tatoeba_path,
        groups_path,
        pairs_path,
        max_pairs=args.max_pairs,
        max_per_word=args.max_per_word,
    )

    manifest = {
        "sources": [
            {"name": "JMdict", "url": JM_DICT_URL, "sha256": _sha256(jmdict_path), "license": "EDRDG JMdict licence"},
            {"name": "Tatoeba Japanese sentences", "url": TATOEBA_URL, "sha256": _sha256(tatoeba_path), "license": "CC BY 2.0 FR"},
            {"name": "文化庁 異字同訓", "url": "https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/sanko/yohorei/index.html", "license": "政府公開資料（出典表示）"},
        ],
        "jmdict_homophone_groups": group_count,
        "tatoeba_training_pairs": pair_count,
        "max_per_word": args.max_per_word,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
