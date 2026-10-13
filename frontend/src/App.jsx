import { useState } from "react";

const EXAMPLES = [
  "How many days of annual leave do I get?",
  "What is the hotel cap per night in metro cities?",
  "Is there a budget for courses and certifications?",
  "What is the CEO's salary?", // not in the handbook: the agent should abstain
];

export default function App() {
  const [q, setQ] = useState("");
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [showTrace, setShowTrace] = useState(false);

  async function ask(question) {
    if (!question.trim()) return;
    setLoading(true); setErr(""); setRes(null);
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!r.ok) throw new Error(`Request failed (${r.status})`);
      setRes(await r.json());
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Handbook Assistant</h1>
      <p className="sub">Agentic RAG: hybrid search, grounded answers with citations, and it says "I don't know" when the handbook is silent.</p>

      <div className="row">
        <input value={q} onChange={(e) => setQ(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && ask(q)}
               placeholder="Ask a policy question..." />
        <button onClick={() => ask(q)} disabled={loading}>{loading ? "Searching..." : "Ask"}</button>
      </div>
      <div className="chips">
        {EXAMPLES.map((e) => (
          <button key={e} className="chip" onClick={() => { setQ(e); ask(e); }}>{e}</button>
        ))}
      </div>

      {err && <p className="error">{err}</p>}

      {res && (
        <section className={res.abstained ? "card abstain" : "card"}>
          <div className="meta">
            <span className={res.abstained ? "tag warn" : "tag ok"}>{res.abstained ? "abstained" : "grounded"}</span>
            <span className="tag">{res.mode}</span>
            <span className="sub">{res.latency_ms} ms</span>
          </div>
          <p className="answer">{res.answer}</p>

          {res.citations.length > 0 && (
            <>
              <h3>Sources</h3>
              {res.citations.map((c) => (
                <div key={c.id} className="cite">
                  <strong>[{c.n}] {c.title} / {c.heading}</strong>
                  <p>{c.snippet}</p>
                </div>
              ))}
            </>
          )}

          <button className="link" onClick={() => setShowTrace(!showTrace)}>
            {showTrace ? "Hide" : "Show"} agent trace
          </button>
          {showTrace && <pre>{JSON.stringify(res.trace, null, 2)}</pre>}
        </section>
      )}
    </main>
  );
}
