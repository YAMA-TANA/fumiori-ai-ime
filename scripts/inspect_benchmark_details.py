import json
from pathlib import Path

def main():
    path = Path("build/human_prefix_benchmark_cpu_v12.json")
    if not path.exists():
        print("Not found:", path)
        return
    data = json.load(open(path, encoding="utf-8"))
    rows = data.get("rows", [])
    print(f"Total rows (benchmark cases): {len(rows)}")
    
    raw_correct = sum(1 for r in rows if r.get("raw_correct"))
    ai_correct = sum(1 for r in rows if r.get("ai_correct"))
    changed = [r for r in rows if r.get("raw_surface") != r.get("ai_surface")]
    recovered = [r for r in rows if not r.get("raw_correct") and r.get("ai_correct")]
    regressed = [r for r in rows if r.get("raw_correct") and not r.get("ai_correct")]
    
    print(f"Mozc raw correct: {raw_correct}/{len(rows)} ({raw_correct/len(rows)*100:.1f}%)")
    print(f"AI rerank correct: {ai_correct}/{len(rows)} ({ai_correct/len(rows)*100:.1f}%)")
    print(f"AI changed: {len(changed)} cases")
    print(f"Recovered (Mozc NG -> AI OK): {len(recovered)} cases")
    print(f"Regressed (Mozc OK -> AI NG): {len(regressed)} cases")
    
    latencies = [r.get("elapsed_ms", 0) for r in rows if r.get("ai_called")]
    if latencies:
        print(f"Avg latency per conversion: {sum(latencies)/len(latencies):.2f} ms (min: {min(latencies):.2f}, max: {max(latencies):.2f})")
    
    print("\n" + "="*80)
    print("DETAILED SAMPLE RECOVERIES BY CATEGORY (Actual Mozc Candidates -> AI Rerank)")
    print("="*80)
    
    by_cat = {}
    for r in recovered:
        cat = r.get("category", "Other")
        by_cat.setdefault(cat, []).append(r)
        
    for cat, items in by_cat.items():
        print(f"\n### Category: {cat} (Recovered: {len(items)} cases)")
        for r in items[:6]:
            reading = r.get("reading")
            mozc_cand = r.get("raw_surface")
            ai_cand = r.get("ai_surface")
            expected = r.get("expected")
            counts = r.get("candidate_counts")
            elapsed = r.get("elapsed_ms")
            traces = r.get("traces", [])
            cand_list = []
            if traces and traces[0].get("explanation"):
                cand_list = [c["text"] for c in traces[0]["explanation"].get("candidates", [])]
            print(f"  - [{r.get('label')}] Form: {r.get('form')} | Reading: '{reading}'")
            print(f"    Mozc Default: '{mozc_cand}' --> AI Rerank: '{ai_cand}' (Expected: '{expected}')")
            print(f"    Candidates (top evaluated): {cand_list[:6]} ... (total {counts} candidates, {elapsed:.1f}ms)")

if __name__ == "__main__":
    main()
