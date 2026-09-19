import subprocess
import time
from pathlib import Path
from client.windows_pipe import rank_once

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "YamatanaAIIME" / "YamatanaAIIME.exe"
PIPE_NAME = r"\\.\pipe\test_dual_encoder_pipe"

# Launch frozen executable in server mode on a test pipe
proc = subprocess.Popen(
    [str(EXE), "--server", "--pipe", PIPE_NAME, "--no-ui"],
    cwd=str(ROOT),
)
try:
    # Wait for pipe to be ready
    time.sleep(2.5)
    
    req = {
        "request_id": "probe-test",
        "inference_trigger": "explicit",
        "preceding_text": "役員が稟議書を",
        "read": "けっさい",
        "candidates": [
            {"id": "c1", "text": "決済", "rank": 1},
            {"id": "c2", "text": "決裁", "rank": 2},
            {"id": "c3", "text": "血清", "rank": 3},
        ],
    }
    
    resp = rank_once(PIPE_NAME, req, timeout_ms=5000)
    print("Full response:", resp)
    candidates = resp.get("candidates", [])
    winner = candidates[0].get("text") or candidates[0].get("word") or candidates[0].get("id")
    print("Winner:", winner)
    first_item = candidates[0]
    assert first_item["id"] == "c2", f"Expected c2, got {first_item}"
    print("FROZEN PIPE SERVER TEST PASSED! Winner is c2 (決裁) with score:", first_item["score"])
finally:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
