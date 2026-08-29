# Baseline iterative tests for EvoEval_creative/28

def baseline_check_0(candidate):
    assert candidate(10001) == 0.15
    assert candidate(10000, 0) == 0
    assert candidate(20000, 0) == 1500.0
    assert candidate(55000, 2500) == 7750.0
    assert candidate(120000, 5000) == 23700.0

def baseline_check_1(candidate):
    assert candidate(9999.5, 0) == 0
