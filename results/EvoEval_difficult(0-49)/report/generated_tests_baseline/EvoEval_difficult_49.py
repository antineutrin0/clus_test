# Baseline iterative tests for EvoEval_difficult/49

def baseline_check_0(candidate):
    assert candidate(100, 100, 10) == -1
    assert candidate(3, 11, 2) == 10
