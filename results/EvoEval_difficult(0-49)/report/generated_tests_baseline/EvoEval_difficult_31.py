# Baseline iterative tests for EvoEval_difficult/31

def baseline_check_0(candidate):
    assert candidate([2, 3], 5) is True
    assert candidate([1, 1], 2) is False
    assert candidate([2, 11], 10) is False
    assert candidate([4, 6], 10) is False
