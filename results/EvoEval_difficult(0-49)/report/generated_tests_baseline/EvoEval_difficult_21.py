# Baseline iterative tests for EvoEval_difficult/21

def baseline_check_0(candidate):
    assert candidate([1.0, 2.0, 4.0, None, 3.0]) == [0.0, 0.33, 1.0, None, 0.67]
