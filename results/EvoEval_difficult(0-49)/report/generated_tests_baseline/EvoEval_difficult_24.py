# Baseline iterative tests for EvoEval_difficult/24

def baseline_check_0(candidate):
    assert candidate(12, 1) == 3
    assert candidate(6, 2) == 2
    assert candidate(12, 3) == -1
