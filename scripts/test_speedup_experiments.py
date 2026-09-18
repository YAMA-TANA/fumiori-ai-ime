import time
from pathlib import Path
import sys
import concurrent.futures
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker

def test_speedup_experiments():
    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    
    print("Initializing OnnxRuriReranker...")
    ranker = OnnxRuriReranker(
        model_paths=[lora3_fp16, lora6_fp16],
        ensemble_weights=[0.25, 0.75],
        prior_w=0.0,
    )
    
    prefix = "役員が稟議書を"
    reading = "けっさい"
    candidates_full = ["決済", "決裁", "けっさい", "決裁者", "血清", "結納", "欠席", "決算"]
    
    query = ranker._build_query(prefix, "")
    
    print("\n" + "="*80)
    print("SPEED IMPROVEMENT EXPERIMENTS & PROFILING")
    print("="*80)
    
    # ---------------------------------------------------------
    # Experiment 1: Candidate count (Batch size) impact
    # ---------------------------------------------------------
    print("\n--- Experiment 1: Impact of Candidate Count (Batch Size) ---")
    for k in [8, 5, 4, 3, 2]:
        cands = candidates_full[:k]
        encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(cands), [f"{prefix}{c}" for c in cands])))
        inputs = {
            "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        }
        # Warmup
        ranker.sessions[0].run(["logits"], inputs)
        
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            # Single session run
            ranker.sessions[0].run(["logits"], inputs)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"Candidates = {k:<2} | Single Model Latency: mean {np.mean(times):.2f} ms (p50: {np.median(times):.2f} ms)")

    # ---------------------------------------------------------
    # Experiment 2: Single Model vs Two-Model Ensemble
    # ---------------------------------------------------------
    print("\n--- Experiment 2: Single Model vs Two-Model Ensemble (k=5) ---")
    cands = candidates_full[:5]
    encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(cands), [f"{prefix}{c}" for c in cands])))
    inputs = {
        "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
    }
    
    # Sequential Ensemble (Current)
    seq_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = [s.run(["logits"], inputs)[0] for s in ranker.sessions]
        seq_times.append((time.perf_counter() - t0) * 1000)
    print(f"Current Sequential Ensemble (2 models): {np.mean(seq_times):.2f} ms")

    # Single Model (LoRA6 alone)
    single_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = ranker.sessions[1].run(["logits"], inputs)[0]
        single_times.append((time.perf_counter() - t0) * 1000)
    print(f"Single Model (LoRA6 only)            : {np.mean(single_times):.2f} ms ({np.mean(seq_times)/np.mean(single_times):.2f}x faster)")

    # Parallel Execution (ThreadPool)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        par_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            f1 = executor.submit(ranker.sessions[0].run, ["logits"], inputs)
            f2 = executor.submit(ranker.sessions[1].run, ["logits"], inputs)
            _ = [f1.result(), f2.result()]
            par_times.append((time.perf_counter() - t0) * 1000)
    print(f"Parallel Ensemble (ThreadPool)       : {np.mean(par_times):.2f} ms ({np.mean(seq_times)/np.mean(par_times):.2f}x faster)")

    # ---------------------------------------------------------
    # Experiment 3: Context Truncation / Max Length impact
    # ---------------------------------------------------------
    print("\n--- Experiment 3: Truncation Max Length (256 vs 128 vs 64) ---")
    for max_l in [256, 128, 64]:
        ranker.tokenizer.enable_truncation(max_length=max_l)
        encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(cands), [f"{prefix}{c}" for c in cands])))
        inputs = {
            "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        }
        tok_len = inputs["input_ids"].shape[1]
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            ranker.sessions[1].run(["logits"], inputs)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"max_length = {max_l:<3} (actual seq_len={tok_len:<2}) | Latency: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    test_speedup_experiments()
