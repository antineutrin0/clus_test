# Baseline iterative tests for EvoEval_creative/0

def baseline_check_0(candidate):
    assert candidate([5.0], "Local") == "5.00"
    assert candidate([5.5], "International") == "37.50"
    assert candidate([1.0], "National") == "7.50"
    assert candidate(["x", 1.0], "Mars") == "15.00"
