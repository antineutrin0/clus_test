# Baseline iterative tests for EvoEval_difficult/35

def baseline_check_0(candidate):
    assert candidate([1, 2, 3], 2) == (2, 1)
    assert candidate([3, 1, 2], 3) == (1, 2)
    assert candidate([10], 2) is None
