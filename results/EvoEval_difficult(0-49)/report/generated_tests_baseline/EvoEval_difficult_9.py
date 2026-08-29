# Baseline iterative tests for EvoEval_difficult/9

def baseline_check_0(candidate):
    assert candidate([]) == []
    assert candidate([0]) == [(0, 0)]
    assert candidate([1, 2, 3, 2, 3, 4, 2]) == [(1, 1), (2, 1), (3, 1), (3, 1), (3, 1), (4, 1), (4, 1)]
