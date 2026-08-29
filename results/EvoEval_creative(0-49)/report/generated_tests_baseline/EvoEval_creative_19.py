# Baseline iterative tests for EvoEval_creative/19

def baseline_check_0(candidate):
    assert candidate([[1, 0], [0, 1]]) == 0
    assert candidate([[2, 1], [1, 1]]) == 2
    assert candidate([[2, 1], [1, 0]]) == 2
