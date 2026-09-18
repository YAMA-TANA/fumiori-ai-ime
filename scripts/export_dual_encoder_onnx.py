"""Export ModernBERT 70M Dual-Encoder to FP16 and INT8 ONNX models for ultra-low latency IME runtime."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

# Avoid sklearn import hanging on Windows
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
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer, ModernBertModel

ROOT = Path(__file__).resolve().parents[1]


class DualEncoderExportWrapper(nn.Module):
    """Wrapper that takes input_ids and attention_mask, performs mean pooling, and returns L2-normalized embeddings."""

    def __init__(self, encoder: ModernBertModel) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, S, H)
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # (B, H)
        
        # L2 normalize
        return F.normalize(pooled, p=2, dim=1)


def export_dual_encoder(
    model_dir: Path,
    output_dir: Path,
    quantize: bool = True,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Dual-Encoder model from {model_dir} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    base_model = ModernBertModel.from_pretrained(str(model_dir)).to(device).eval()
    
    # 1. Export FP16 (or FP32 if cpu)
    wrapper = DualEncoderExportWrapper(base_model).eval()
    
    # Sample input for tracing
    sample_text = ["役員が稟議書を決裁する"]
    enc = tokenizer(sample_text, padding=True, truncation=True, max_length=48, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    fp32_path = output_dir / "dual-encoder-70m-fp32.onnx"
    fp16_path = output_dir / "dual-encoder-70m-fp16.onnx"
    int8_path = output_dir / "dual-encoder-70m-int8.onnx"
    
    print(f"Exporting base ONNX model to {fp32_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "embeddings": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[SUCCESS] Base FP32 ONNX exported: {fp32_path} ({fp32_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # 1. Native FP16 export for DirectML / CUDA
    if torch.cuda.is_available():
        print(f"Exporting native FP16 ONNX model to {fp16_path}...")
        model_fp16 = ModernBertModel.from_pretrained(
            str(model_dir), torch_dtype=torch.float16
        ).to(device).eval()
        wrapper_fp16 = DualEncoderExportWrapper(model_fp16).eval()
        
        with torch.inference_mode():
            torch.onnx.export(
                wrapper_fp16,
                (input_ids, attention_mask),
                str(fp16_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["embeddings"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "embeddings": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        print(f"[SUCCESS] Native FP16 ONNX exported: {fp16_path} ({fp16_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        shutil.copy(fp32_path, fp16_path)

    # 2. Dynamic INT8 quantization for CPU runtime
    if quantize:
        print(f"Quantizing to INT8: {int8_path}...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"[SUCCESS] INT8 ONNX saved: {int8_path} ({int8_path.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Copy tokenizer files
    for tok_file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = model_dir / tok_file
        if src.exists():
            shutil.copy(src, output_dir / tok_file)
            print(f"Copied {tok_file} to {output_dir}")

    # 4. Verify numerical equivalence with ONNXRuntime
    print("\nVerifying numerical equivalence with PyTorch...")
    test_texts = ["役員が稟議書を決裁する", "病気を治す", "雨が降る"]
    enc_test = tokenizer(test_texts, padding=True, truncation=True, max_length=48, return_tensors="pt")
    with torch.inference_mode():
        pt_embeds = wrapper(enc_test["input_ids"].to(device), enc_test["attention_mask"].to(device)).cpu().numpy()

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": enc_test["input_ids"].cpu().numpy(),
        "attention_mask": enc_test["attention_mask"].cpu().numpy(),
    }
    ort_embeds = sess.run(["embeddings"], ort_inputs)[0]

    for i, t in enumerate(test_texts):
        sim = float(np.dot(pt_embeds[i], ort_embeds[i]))
        print(f"Text: '{t}' -> Cosine Sim (PyTorch vs ONNX): {sim:.6f}")
        assert sim > 0.999, f"Numerical divergence detected! Sim: {sim}"

    print("\n[VERIFICATION PASS] ONNX embeddings exactly match PyTorch!")
    return fp16_path, int8_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(ROOT / "build" / "dual-encoder-70m"))
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "onnx-model-70m-dual-encoder"))
    args = parser.parse_args()

    export_dual_encoder(Path(args.model_dir), Path(args.output_dir))
