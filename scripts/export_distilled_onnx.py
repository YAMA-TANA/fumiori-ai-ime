"""Export and quantize distilled student Ruri reranker for Windows runtime (DirectML FP16 + CPU INT8)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

# Avoid an optional sklearn/pandas import chain in Transformers 4.57 on
# Windows.  Exporting a model never uses the sklearn generation helpers.
_find_spec = importlib.util.find_spec


def _find_spec_without_sklearn(name: str, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_sklearn

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ModernBertForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_tokenizer(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, fix_mistral_regex=False)
    except (TypeError, ValueError):
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def export_native_fp16(model_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(model_dir)

    try:
        model = ModernBertForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float16, num_labels=1
        ).to(device).eval()
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float16, trust_remote_code=True
        ).to(device).eval()

    wrapper = LogitsOnly(model).eval()
    pairs = tokenizer(
        ["文書方針: 医学文書。\n文脈「医師が病気を____」に適切な表記を選ぶ。"],
        ["医師が病気を治す"],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = pairs["input_ids"].to(device)
    attention_mask = pairs["attention_mask"].to(device)
    fp16_path = output_dir / "ruri-ime-fp16.onnx"

    print(f"Exporting native FP16 ONNX to {fp16_path}...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(fp16_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    size_mb = fp16_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Exported native FP16 model: {fp16_path} ({size_mb:.2f} MB)")
    return fp16_path


def export_model(model_dir: Path, output_dir: Path, quantize: bool = True, reuse_fp32: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(model_dir)

    try:
        model = ModernBertForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float32, num_labels=1
        ).cpu().eval()
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float32, trust_remote_code=True
        ).cpu().eval()

    wrapper = LogitsOnly(model).eval()

    pairs = tokenizer(
        ["文書方針: 医学文書。\n文脈「医師が病気を____」に適切な表記を選ぶ。"],
        ["医師が病気を治す"],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    float_path = output_dir / "ruri-ime-fp32.onnx"
    if not reuse_fp32 or not float_path.exists():
        print(f"Exporting FP32 ONNX graph to {float_path}...", flush=True)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (pairs["input_ids"], pairs["attention_mask"]),
                str(float_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        print(f"FP32 ONNX size: {float_path.stat().st_size / (1024 * 1024):.2f} MB")

    fp16_path = output_dir / "ruri-ime-fp16.onnx"
    final_path = float_path

    if quantize:
        final_path = output_dir / "ruri-ime-int8.onnx"
        print(f"Quantizing to Dynamic INT8 with MatMulConstBOnly: {final_path}...", flush=True)
        quantize_dynamic(
            model_input=str(float_path),
            model_output=str(final_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            extra_options={"MatMulConstBOnly": True},
        )
        int8_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Quantized INT8 ONNX size: {int8_mb:.2f} MB")

    # Verification
    print("Verifying ONNX vs PyTorch output logits...", flush=True)
    session = ort.InferenceSession(str(final_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(
        ["logits"],
        {
            "input_ids": pairs["input_ids"].numpy().astype(np.int64),
            "attention_mask": pairs["attention_mask"].numpy().astype(np.int64),
        },
    )[0]
    with torch.inference_mode():
        torch_logits = wrapper(pairs["input_ids"], pairs["attention_mask"]).numpy()

    report = {
        "model": str(final_path),
        "int8_bytes": final_path.stat().st_size,
        "int8_mb": round(final_path.stat().st_size / (1024 * 1024), 2),
        "fp32_bytes": float_path.stat().st_size if float_path.exists() else None,
        "fp32_mb": round(float_path.stat().st_size / (1024 * 1024), 2) if float_path.exists() else None,
        "fp16_bytes": fp16_path.stat().st_size if fp16_path.exists() else None,
        "fp16_mb": round(fp16_path.stat().st_size / (1024 * 1024), 2) if fp16_path.exists() else None,
        "torch_logits": torch_logits.reshape(-1).tolist(),
        "onnx_logits": np.asarray(ort_logits).reshape(-1).tolist(),
        "max_absolute_error": float(np.max(np.abs(torch_logits - ort_logits))),
    }
    verification_path = output_dir / "verification.json"
    verification_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "ruri-v3-70m-ime-distilled",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "onnx-model-70m",
    )
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--reuse-fp32", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()

    # Export both FP16 (GPU) and INT8 (CPU)
    export_native_fp16(model_dir, output_dir)
    export_model(model_dir, output_dir, not args.no_quantize, reuse_fp32=args.reuse_fp32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
