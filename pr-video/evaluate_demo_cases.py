from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_70M = ROOT / "build" / "onnx-model-70m" / "ruri-ime-int8.onnx"
MODEL_310M = ROOT / "build" / "onnx-model" / "ruri-ime-int8.onnx"
MODEL = MODEL_70M if MODEL_70M.exists() else MODEL_310M
sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker


CASES = [
    {
        "label": "music",
        "preceding_text": "この曲の",
        "following_text": "は、サビで一オクターブ上がる。",
        "read": "しゅせんりつ",
        "candidates": ["主戦率", "主旋律", "主選率", "主線率"],
    },
    {
        "label": "medical",
        "preceding_text": "医師の治療で長年の病気を",
        "following_text": "ことができた。",
        "read": "なおす",
        "candidates": ["直す", "治す", "なおす"],
    },
    {
        "label": "legal",
        "preceding_text": "改正された法律は来月から",
        "following_text": "される。",
        "read": "しこう",
        "candidates": ["思考", "試行", "志向", "指向", "施行"],
    },
    {
        "label": "experiment",
        "preceding_text": "新しいアルゴリズムを本番環境で",
        "following_text": "して性能を確かめる。",
        "read": "しこう",
        "candidates": ["思考", "施行", "志向", "指向", "試行"],
    },
    {
        "label": "antenna",
        "preceding_text": "衛星アンテナの",
        "following_text": "性を測定する。",
        "read": "しこう",
        "candidates": ["思考", "施行", "試行", "志向", "指向"],
    },
    {
        "label": "garden",
        "preceding_text": "庭に咲いた美しい",
        "following_text": "を眺める。",
        "read": "はな",
        "candidates": ["鼻", "花", "はな"],
    },
]


def main() -> None:
    settings = {
        "context_enabled": True,
        "context_chars": 128,
        "document_domain": "general",
        "custom_instruction": "",
        "lexical_grounding": True,
        "compute_mode": "cpu",
    }
    ranker = OnnxRuriReranker(settings=settings, model_path=MODEL)
    results = []
    for index, case in enumerate(CASES, start=1):
        request = {
            "request_id": f"demo-{index}",
            "preceding_text": case["preceding_text"],
            "following_text": case["following_text"],
            "read": case["read"],
            "candidates": [
                {"id": f"c{candidate_index}", "text": text, "rank": candidate_index}
                for candidate_index, text in enumerate(case["candidates"], start=1)
            ],
        }
        response = ranker.rank(request)
        by_id = {item["id"]: item for item in response["candidates"]}
        ranked = sorted(
            (
                {
                    "text": item["text"],
                    "original_rank": item["rank"],
                    "ai_rank": by_id[item["id"]]["rank"],
                    "score": round(by_id[item["id"]]["score"], 4),
                }
                for item in request["candidates"]
            ),
            key=lambda item: item["ai_rank"],
        )
        results.append(
            {
                "label": case["label"],
                "context": case["preceding_text"] + "____" + case["following_text"],
                "read": case["read"],
                "latency_ms": ranker.last_latency_ms,
                "ranked": ranked,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
