from evals.run_eval import run


def test_quality_gate_passes():
    result = run(k=3)
    assert not result["failures"], result["failures"]
