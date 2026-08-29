# Baseline iterative tests for EvoEval_creative/34

def baseline_check_0(candidate):
    assert candidate(23, 12) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 0]

def baseline_check_1(candidate):
    assert candidate(10, 8) == [0, 10, 20, 30, 40, 50, 60, 100, 90, 80]
