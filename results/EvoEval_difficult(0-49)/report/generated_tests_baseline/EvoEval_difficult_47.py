# Baseline iterative tests for EvoEval_difficult/47

def baseline_check_0(candidate):
    assert candidate([1, 10, 20, 30], [2, 1, 1, 2]) == 15.0
    assert candidate([1, 2, 3], [2, 1, 1]) == 1

def baseline_check_1(candidate):
    assert candidate([0], [0]) == 0
