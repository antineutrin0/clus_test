# Baseline iterative tests for EvoEval_difficult/8

def baseline_check_0(candidate):
    assert candidate([]) == (0, 1, 0, 1)
    assert candidate([-1, -2, 3, 4]) == (2, -3, 2, -8)
