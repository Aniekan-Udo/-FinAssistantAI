"""Evaluate a running FinAssist server: tool-routing accuracy, knowledge-base answer accuracy and latency.

Usage:
    python eval/run_eval.py [BASE_URL] [SET]   # defaults: http://localhost:8000, eval_set
    SET is eval_set (used while fixing bugs) or holdout_set (never tuned against)

Queries are paced to stay inside the Groq free-tier rate limit (each query makes ~3 LLM calls).
Results are printed and saved to eval/results_<SET>.json.
"""
import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from urllib import request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
SET_NAME = sys.argv[2] if len(sys.argv) > 2 else "eval_set"  # or holdout_set
EVAL_DIR = Path(__file__).parent
PACE_SECONDS = 8


def ask(query: str, session_id: str) -> tuple[dict, float]:
    body = json.dumps({"query": query, "session_id": session_id}).encode()
    req = request.Request(f"{BASE_URL}/query", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data, time.perf_counter() - start


def normalize(text: str) -> str:
    """Lowercase and drop spaces before % so "1.5 %" (narrow no-break space) matches "1.5%"."""
    return re.sub(r"\s+%", "%", text.replace("\u202f", " ").replace("\xa0", " ")).lower()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def main():
    cases = [json.loads(line) for line in (EVAL_DIR / f"{SET_NAME}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    session_id = uuid.uuid4().hex
    rows = []

    for i, case in enumerate(cases, 1):
        try:
            data, seconds = ask(case["query"], session_id)
            error = None
        except Exception as e:
            data, seconds, error = {}, None, str(e)

        answer = data.get("response", "")
        routed_ok = data.get("tool_used") == case["tool"]
        answer_ok = all(normalize(fact) in normalize(answer) for fact in case["expect"]) if "expect" in case else None
        rows.append({**case, "tool_used": data.get("tool_used"), "routed_ok": routed_ok,
                     "answer_ok": answer_ok, "seconds": seconds, "error": error, "answer": answer})
        mark = "ok " if routed_ok and answer_ok is not False else "BAD"
        print(f"[{i:2}/{len(cases)}] {mark} {seconds or 0:5.1f}s  want={case['tool']!r} got={data.get('tool_used')!r}  {case['query']}")
        time.sleep(PACE_SECONDS)

    latencies = [r["seconds"] for r in rows if r["seconds"] is not None]
    kb_rows = [r for r in rows if r["answer_ok"] is not None]
    summary = {
        "base_url": BASE_URL,
        "set": SET_NAME,
        "cases": len(rows),
        "errors": sum(r["error"] is not None for r in rows),
        "routing_accuracy": sum(r["routed_ok"] for r in rows) / len(rows),
        "kb_answer_accuracy": sum(r["answer_ok"] for r in kb_rows) / len(kb_rows) if kb_rows else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
    }
    (EVAL_DIR / f"results_{SET_NAME}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
