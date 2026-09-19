import time
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker

def profile_detailed():
    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    
    ranker = OnnxRuriReranker(
        model_paths=[lora3_fp16, lora6_fp16],
        ensemble_weights=[0.25, 0.75],
        prior_w=0.0,
    )
    
    sample_cases = [
        {
            "prefix": "役員が稟議書を",
            "reading": "けっさい",
            "candidates": ["決済", "決裁", "けっさい", "決裁者", "血清"],
        },
        {
            "prefix": "新しい暗号化アルゴリズムをC++言語で効率的に",
            "reading": "じっそう",
            "candidates": ["実相", "実装", "じっそう", "実相寺", "実装例"],
        },
        {
            "prefix": "オンプレミスの旧サーバーからクラウド環境へ",
            "reading": "いこう",
            "candidates": ["以降", "移行", "意向", "威光", "イコウ"],
        }
    ]
    
    # Warmup
    for c in sample_cases:
        req = {
            "request_id": "warmup",
            "preceding_text": c["prefix"],
            "read": c["reading"],
            "candidates": [{"id": f"c{i}", "text": cand, "rank": i+1} for i, cand in enumerate(c["candidates"])]
        }
        ranker.rank(req)
        
    print("=" * 80)
    print("DETAILED LATENCY BREAKDOWN PROFILING (10 ITERATIONS)")
    print("=" * 80)
    
    tokenize_times = []
    onnx_lora3_times = []
    onnx_lora6_times = []
    postprocess_times = []
    total_times = []
    seq_lengths = []
    
    for _ in range(10):
        for c in sample_cases:
            candidates = c["candidates"]
            prefix = c["prefix"]
            reading = c["reading"]
            
            t0 = time.perf_counter()
            query = ranker._build_query(prefix, "")
            documents = [f"{prefix}{cand}" for cand in candidates]
            
            # 1. Tokenization
            t_tok_start = time.perf_counter()
            encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(candidates), documents)))
            inputs = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
            }
            t_tok_end = time.perf_counter()
            tokenize_times.append((t_tok_end - t_tok_start) * 1000)
            seq_lengths.append(inputs["input_ids"].shape[1])
            
            # 2. ONNX Run Model 1 (LoRA3)
            t_m1_start = time.perf_counter()
            logits1 = ranker.sessions[0].run(["logits"], inputs)[0]
            t_m1_end = time.perf_counter()
            onnx_lora3_times.append((t_m1_end - t_m1_start) * 1000)
            
            # 3. ONNX Run Model 2 (LoRA6)
            t_m2_start = time.perf_counter()
            logits2 = ranker.sessions[1].run(["logits"], inputs)[0]
            t_m2_end = time.perf_counter()
            onnx_lora6_times.append((t_m2_end - t_m2_start) * 1000)
            
            # 4. Post-processing
            t_post_start = time.perf_counter()
            logits = 0.25 * logits1.reshape(-1) + 0.75 * logits2.reshape(-1)
            # simulated scoring logic
            scores = []
            for i, cand in enumerate(candidates):
                raw = float(logits[i])
                scores.append(raw)
            t_post_end = time.perf_counter()
            postprocess_times.append((t_post_end - t_post_start) * 1000)
            
            total_times.append((t_post_end - t0) * 1000)

    print(f"Average Token Sequence Length: {np.mean(seq_lengths):.1f} tokens (max_length padded to 256)")
    print(f"1. Tokenization Time        : {np.mean(tokenize_times):.2f} ms ({np.mean(tokenize_times)/np.mean(total_times)*100:.1f}%)")
    print(f"2. ONNX Session 1 (LoRA3)    : {np.mean(onnx_lora3_times):.2f} ms ({np.mean(onnx_lora3_times)/np.mean(total_times)*100:.1f}%)")
    print(f"3. ONNX Session 2 (LoRA6)    : {np.mean(onnx_lora6_times):.2f} ms ({np.mean(onnx_lora6_times)/np.mean(total_times)*100:.1f}%)")
    print(f"4. Post-processing Time      : {np.mean(postprocess_times):.2f} ms ({np.mean(postprocess_times)/np.mean(total_times)*100:.1f}%)")
    print(f"--------------------------------------------------------------------------------")
    print(f"Total Latency per Conversion : {np.mean(total_times):.2f} ms (p50: {np.median(total_times):.2f} ms, min: {np.min(total_times):.2f} ms)")

if __name__ == "__main__":
    profile_detailed()
