from app import agent
from app.text import coverage


def test_hybrid_search_finds_remote_policy():
    hits = agent.get_retriever().search("work from home days", 3)
    assert hits[0].chunk.source == "remote-work"


def test_answer_has_fact_and_citation():
    out = agent.answer("How many days of annual leave do I get?")
    assert "24" in out["answer"]
    assert out["citations"][0]["source"] == "leave-policy"
    assert not out["abstained"]


def test_abstains_when_not_in_docs():
    out = agent.answer("What is the CEO's salary?")
    assert out["abstained"] and out["citations"] == []


def test_coverage_bounds():
    assert coverage("annual leave", "Annual leave is 24 days") == 1.0
    assert coverage("stock ticker", "Annual leave is 24 days") == 0.0
