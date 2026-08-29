# Baseline iterative tests for EvoEval_difficult/25

def baseline_check_0(candidate):
    assert candidate(8) == [(2, 3)]
    assert candidate(25) == [(5, 2)]
    assert candidate(2) == [(2, 1)]
