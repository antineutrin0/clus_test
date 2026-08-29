# Baseline iterative tests for EvoEval_difficult/42

def baseline_check_0(candidate):
    assert candidate([1, 2, 3], [2]) == [2, 2, 4]
    assert candidate([1], [2, "3"]) == "Ignore list should only contain integers"
