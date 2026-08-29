# Baseline iterative tests for EvoEval_difficult/30

def baseline_check_0(candidate):
    data = [10, 11, 1, -7, 13, 2, 0, 3]
    assert candidate(data) == [(1, 2), (2, 5), (3, 7)]
