from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranker.onnx_ranker import OnnxRuriReranker

def test_live_reranking():
    # LoRA3 + LoRA6 calibrated ensemble (v2.0.6-beta / current runtime)
    lora3_fp16 = Path("build/onnx-model-70m-lora3-20260909/ruri-ime-fp16.onnx")
    lora6_fp16 = Path("build/onnx-model-70m-lora6-preceding-only-20260915/ruri-ime-fp16.onnx")
    
    models = [lora3_fp16, lora6_fp16] if lora3_fp16.exists() and lora6_fp16.exists() else None
    
    print(f"Loading Ranker with models: {models}...")
    t0 = time.time()
    ranker = OnnxRuriReranker(
        model_paths=models,
        ensemble_weights=[0.25, 0.75] if models else None,
        prior_w=0.0 if models else None,
    )
    print(f"Ranker initialized in {(time.time() - t0)*1000:.1f}ms")
    print("Execution Providers:", [s.get_providers() for s in ranker.sessions])
    
    test_cases = [
        {
            "label": "業務・稟議決裁",
            "prefix": "役員が稟議書を",
            "reading": "けっさい",
            "candidates": ["決済", "決裁", "けっさい", "決裁者", "血清"],
            "mozc_top": "決済",
            "expected": "決裁",
        },
        {
            "label": "日常・体の一部（鼻 vs 花）",
            "prefix": "彼の顔の真ん中にある",
            "reading": "はな",
            "candidates": ["花", "鼻", "華", "端", "ハナ"],
            "mozc_top": "花",
            "expected": "鼻",
        },
        {
            "label": "IT技術・システム移行",
            "prefix": "オンプレミスの旧サーバーからクラウド環境へ",
            "reading": "いこう",
            "candidates": ["以降", "移行", "意向", "威光", "イコウ"],
            "mozc_top": "以降",
            "expected": "移行",
        },
        {
            "label": "法律・施行",
            "prefix": "改正された新制度が来週から",
            "reading": "しこう",
            "candidates": ["思考", "施行", "試行", "指向", "志向"],
            "mozc_top": "思考",
            "expected": "施行",
        },
        {
            "label": "医療・病気を治す",
            "prefix": "処方された抗生物質で病気を",
            "reading": "なおす",
            "candidates": ["直す", "治す", "猶す", "なおす"],
            "mozc_top": "直す",
            "expected": "治す",
        },
        {
            "label": "日常・天気の雨",
            "prefix": "外は暗くなり大粒の雨が",
            "reading": "ふる",
            "candidates": ["フル", "降る", "振る", "狂る", "ふる"],
            "mozc_top": "フル",
            "expected": "降る",
        },
        {
            "label": "IT技術・アルゴリズム実装",
            "prefix": "新しい暗号化アルゴリズムをC++言語で効率的に",
            "reading": "じっそう",
            "candidates": ["実相", "実装", "じっそう", "実相寺"],
            "mozc_top": "実相",
            "expected": "実装",
        },
        {
            "label": "製造・大型機械",
            "prefix": "組み立てラインに設置された精密な大型",
            "reading": "きかい",
            "candidates": ["機会", "機械", "器械", "奇怪", "きかい"],
            "mozc_top": "機会",
            "expected": "機械",
        }
    ]
    
    print("\n" + "="*80)
    print("LIVE RUN RESULTS (Testing individual conversion requests via ranker.rank)")
    print("="*80)
    
    latencies = []
    success_count = 0
    
    for i, tc in enumerate(test_cases):
        req = {
            "request_id": f"req-{i}",
            "inference_trigger": "explicit",
            "preceding_text": tc["prefix"],
            "following_text": "",
            "read": tc["reading"],
            "candidates": [
                {"id": f"c{j}", "text": cand, "rank": j+1}
                for j, cand in enumerate(tc["candidates"])
            ]
        }
        
        t1 = time.time()
        res = ranker.rank(req)
        lat = (time.time() - t1) * 1000
        latencies.append(lat)
        
        cand_map = {c["id"]: c["text"] for c in req["candidates"]}
        ranked_texts = [cand_map[c["id"]] for c in res["candidates"]]
        top_cand = ranked_texts[0]
        ok = (top_cand == tc["expected"])
        if ok:
            success_count += 1
            
        print(f"\n[{i+1}] {tc['label']} (Latency: {lat:.1f}ms)")
        print(f"    直前文脈: 「{tc['prefix']}」 + 読み: `{tc['reading']}`")
        print(f"    Mozc初期1位: 「{tc['mozc_top']}」")
        print(f"    AIリランク1位: 「{top_cand}」 {'[OK]' if ok else '[NG]'} (期待値: 「{tc['expected']}」)")
        print(f"    候補順位: {ranked_texts}")

    print("\n" + "="*80)
    print(f"Summary: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%) Correct")
    print(f"Avg Latency: {sum(latencies)/len(latencies):.2f} ms (Min: {min(latencies):.2f} ms, Max: {max(latencies):.2f} ms)")
    print("="*80)

if __name__ == "__main__":
    test_live_reranking()
