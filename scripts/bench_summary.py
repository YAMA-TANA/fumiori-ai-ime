import json

def main():
    d = json.load(open("build/query_variant_benchmark_120_ensemble.json", encoding="utf-8"))
    for r in d["results"]:
        print(f"Variant: {r['variant']:<8} | Acc: {r['accuracy']}% | Mean Latency: {r['latency_ms']['mean']}ms | p50: {r['latency_ms']['p50']}ms | p95: {r['latency_ms']['p95']}ms")

if __name__ == "__main__":
    main()
