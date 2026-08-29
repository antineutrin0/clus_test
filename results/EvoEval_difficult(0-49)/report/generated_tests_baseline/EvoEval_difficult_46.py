# Baseline iterative tests for EvoEval_difficult/46

def baseline_check_0(candidate):
    assert candidate(-1, -1) == "Invalid input"
    assert candidate(0, 0) == 0
    assert candidate(1, 1) == 0
    assert candidate(2, 1) == 0
    assert candidate(3, 1) == 2
    assert candidate(3, 2) == 0
    assert candidate(4, 1) == 2
    assert candidate(5, 1) == 4
    assert candidate(6, 1) == 8

def baseline_check_1(candidate):
    assert candidate(0, -5) == 0
    assert candidate(1, -5) == 0
    assert candidate(2, 0) == 2
