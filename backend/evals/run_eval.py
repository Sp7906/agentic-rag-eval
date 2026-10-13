"""Evaluation harness: retrieval quality, answer correctness, citation precision, abstention.

Usage:  python -m evals.run_eval [--k 3] [--offline]
Exits with code 1 if any metric falls below its threshold (used as a CI quality gate).
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
THRESHOLDS = {"hit_at_k": 0.85, "mrr": 0.75, "answer_match": 0.80,
              "citation_precision": 0.85, "abstain_accuracy": 0.90, "false_abstain_rate": 0.10}
LOWER_IS_BETTER = {"false_abstain_rate"}


def _norm(s: str) -> str:
    return s.lower().replace(",", "")


def run(k: int = 3, dataset: Path = HERE / "dataset.jsonl") -> dict:
    from app import agent  # imported late so --offline can set env first
    from app.text import tokens

    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_q, hits, rr, match, cite_p, faith, lat = [], [], [], [], [], [], []
    abst_ok, false_abst = [], []

    for r in rows:
        out = agent.answer(r["question"], k=k)
        srcs = [x["source"] for x in out["retrieved"]]
        lat.append(out["latency_ms"])
        rec = {"id": r["id"], "question": r["question"], "answerable": r["answerable"],
               "abstained": out["abstained"], "answer": out["answer"], "retrieved": srcs}
        if r["answerable"]:
            exp = r["expected_source"]
            hit = exp in srcs
            hits.append(hit)
            rr.append(1 / (srcs.index(exp) + 1) if hit else 0.0)
            false_abst.append(out["abstained"])
            ok = (not out["abstained"]) and any(_norm(p) in _norm(out["answer"]) for p in r["expected_phrases"])
            match.append(ok)
            if not out["abstained"] and out["citations"]:
                cite_p.append(sum(c["source"] == exp for c in out["citations"]) / len(out["citations"]))
                ctx = set(tokens(" ".join(c["snippet"] for c in out["citations"])))
                ans = tokens(out["answer"])
                faith.append(sum(t in ctx for t in ans) / len(ans) if ans else 0.0)
            rec.update(hit=hit, answer_match=ok)
        else:
            abst_ok.append(out["abstained"])
            rec.update(correct_abstain=out["abstained"])
        per_q.append(rec)

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    metrics = {
        "hit_at_k": mean(hits), "mrr": mean(rr), "answer_match": mean(match),
        "citation_precision": mean(cite_p), "faithfulness": mean(faith),
        "abstain_accuracy": mean(abst_ok), "false_abstain_rate": mean(false_abst),
        "p50_latency_ms": round(statistics.median(lat), 1), "k": k, "n_questions": len(rows),
    }
    failures = [f"{m}={metrics[m]} (needs {'<=' if m in LOWER_IS_BETTER else '>='} {t})"
                for m, t in THRESHOLDS.items()
                if (metrics[m] > t if m in LOWER_IS_BETTER else metrics[m] < t)]
    return {"metrics": metrics, "failures": failures, "questions": per_q}


def write_reports(result: dict) -> None:
    (HERE / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    m = result["metrics"]
    lines = ["# Evaluation report", "", "| Metric | Value | Threshold |", "|---|---|---|"]
    for name, t in THRESHOLDS.items():
        lines.append(f"| {name} | {m[name]} | {'<=' if name in LOWER_IS_BETTER else '>='} {t} |")
    lines += [f"| faithfulness | {m['faithfulness']} | - |", f"| p50_latency_ms | {m['p50_latency_ms']} | - |",
              "", "## Failed questions", ""]
    bad = [q for q in result["questions"]
           if (q["answerable"] and not (q.get("hit") and q.get("answer_match")))
           or (not q["answerable"] and not q["correct_abstain"])]
    lines += [f"- {q['id']}: {q['question']} -> {q['answer']}" for q in bad] or ["None"]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--offline", action="store_true", help="ignore Azure credentials")
    args = ap.parse_args()
    if args.offline:
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
    result = run(args.k)
    write_reports(result)
    print(json.dumps(result["metrics"], indent=2))
    if result["failures"]:
        print("\nQUALITY GATE FAILED:\n  " + "\n  ".join(result["failures"]))
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
