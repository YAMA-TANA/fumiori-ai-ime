# Yamatana AI IME PR video

- Format: 9:16 vertical, 720×1280, 30 fps, 25 seconds
- Core claim: the same reading is converted differently from context
- Evidence: every shown ranking was verified against the repository's local INT8 ONNX model

## Timeline

1. 0.0–2.6s — Hook: Japanese IMEs still do not truly read context.
2. 2.6–8.0s — Live-style demo: `しこう` in a legal sentence. `施行` moves from fifth to first in 111 ms.
3. 8.0–13.9s — The same reading produces `施行`, `試行`, or `指向` depending on context.
4. 13.9–18.2s — Hard case: `主戦` + `率` is recombined and corrected to `主旋律`.
5. 18.2–21.8s — Privacy: inference, dictionaries, and context stay on the PC.
6. 21.8–25.0s — Product lockup and GitHub call to action.

## Verified model results

- `改正された法律は来月から____される。` → `施行` (original candidate rank 5 → AI rank 1)
- `新しいアルゴリズムを本番環境で____して性能を確かめる。` → `試行` (5 → 1)
- `衛星アンテナの____性を測定する。` → `指向` (5 → 1)
- `この曲の____は、サビで一オクターブ上がる。` → `主旋律` (2 → 1)
- `医師の治療で長年の病気を____ことができた。` → `治す` (2 → 1)

The displayed 111 ms figure is the measured CPU latency for the five-candidate legal case after model warm-up on the development machine.
