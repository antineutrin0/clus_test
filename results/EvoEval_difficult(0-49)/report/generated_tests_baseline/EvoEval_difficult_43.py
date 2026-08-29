# Baseline iterative tests for EvoEval_difficult/43

def baseline_check_0(candidate):
    assert candidate([], 0) is True
    assert candidate([1], 2) is False
    assert candidate([0], 1) is True
    assert candidate([1, -1, 2], 2) is True
    assert candidate([1, 3, 5, 0], 2) is False
