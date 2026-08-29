# Baseline iterative tests for EvoEval_creative/42

def baseline_check_0(candidate):
    assert candidate([4]) == (False, None)
    assert candidate([5, 1]) == (True, 1)
    assert candidate([5]) == (True, -1)
