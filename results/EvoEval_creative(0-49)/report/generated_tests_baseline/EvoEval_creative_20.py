# Baseline iterative tests for EvoEval_creative/20

def baseline_check_0(candidate):
    assert candidate(0) == []
    assert candidate(1) == ["1", "1"]
    assert candidate(2) == ["1", "1,2", "2,1", "1"]
