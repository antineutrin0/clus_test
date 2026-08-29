# Baseline iterative tests for EvoEval_difficult/40

def baseline_check_0(candidate):
    assert candidate([], 0) == -1
    assert candidate([0, 0, 0], 0) == 1
    assert candidate([0, 1, 2], 3) == 1
    assert candidate([1, 2, 4], 6) == -1
    assert candidate([0, 1, 2, 2], 4) == 1
    assert candidate([0, 1, 2, 3, 5], 5) == 1
    assert candidate([0, 1, 1, 2, 3, 4], 5) == 3

def baseline_check_1(candidate):
    assert candidate([1, -2, 0, 1, 2], 0) == 2
