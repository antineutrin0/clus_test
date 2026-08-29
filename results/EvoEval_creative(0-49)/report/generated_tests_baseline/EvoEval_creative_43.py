# Baseline iterative tests for EvoEval_creative/43

def baseline_check_0(candidate):
    assert candidate(1.0, 3.0, 1.0, 1.0, 1.0) == 0.97
    assert candidate(1.0, 1.0, 1.0, 2.0, 1.0) == 0.8
    assert candidate(1.0, 1.0, 0.5, 1.0, 5.0) == 0.0
    assert candidate(2.0, 1.0, 0.9, 1.0, 5.0) == "Insufficient fuel"
