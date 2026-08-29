# Baseline iterative tests for EvoEval_difficult/32

def baseline_check_0(candidate):
    assert candidate([-5, 16], (0.0, 0.5)) == 0.31
    assert candidate([1, -8, 16], (0.0, 0.5)) is None

def baseline_check_1(candidate):
    assert candidate([-0.00005, 1], (0.0, 0.0001)) == 0.0
    assert candidate([0, 1], (0.0, 1.0)) is None
