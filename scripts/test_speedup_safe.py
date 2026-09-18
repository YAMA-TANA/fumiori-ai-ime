import traceback
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import time
    import numpy as np
    from ranker.onnx_ranker import OnnxRuriReranker

    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    
    print("Initializing OnnxRuriReranker...", flush=True)
    ranker = OnnxRuriReranker(
        model_paths=[lora3_fp16, lora6_fp16],
        ensemble_weights=[0.25, 0.75],
        prior_w=0.0,
    )
    print("Initialized successfully!", flush=True)
    
    prefix = "役員が稟議書を"
    candidates_full = ["決済", "決裁", "けっさい", "決裁者", "血清", "結納", "欠席", "決算"]
    query = ranker._build_query(prefix, "")

    # 1. Candidate count
    for k in [8, 5, 3]:
        cands = candidates_full[:k]
        encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(cands), [f"{prefix}{c}" for c in cands])))
        inputs = {
            "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        }
        ranker.sessions[0].run(["logits"], inputs)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ranker.sessions[0].run(["logits"], inputs)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"Candidates = {k} | Single Model Latency: {np.mean(times):.2f} ms", flush=True)

    # 2. Sequential vs Single
    cands = candidates_full[:5]
    encodings = ranker.tokenizer.encode_batch(list(zip([query]*len(cands), [f"{prefix}{c}" for c in cands])))
    inputs = {
        "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
    }
    
    seq_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = [s.run(["logits"], inputs)[0] for s in ranker.sessions]
        seq_times.append((time.perf_counter() - t0) * 1000)
    print(f"Sequential Ensemble (2 models): {np.mean(seq_times):.2f} ms", flush=True)

    single_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = ranker.sessions[1].run(["logits"], inputs)[0]
        single_times.append((time.perf_counter() - t0) * 1000)
    print(f"Single Model (LoRA6): {np.mean(single_times):.2f} ms", flush=True)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
