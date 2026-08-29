# Baseline iterative tests for EvoEval_creative/4

def baseline_check_0(candidate):
    r = candidate([])
    assert r == 0
    assert type(r) is int

    assert candidate([0]) == 1
    assert candidate([1, 2]) == 2
    assert candidate([1, 1]) == 1
