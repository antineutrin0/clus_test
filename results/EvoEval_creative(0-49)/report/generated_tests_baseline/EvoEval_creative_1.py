# Baseline iterative tests for EvoEval_creative/1

def baseline_check_0(candidate):
    assert candidate([5.0, 6.0, "x"], "Local") == 25.0
    assert candidate([1.0], "National") == 7.5
    assert candidate([1.0], "International") == 12.5
    assert candidate([1.0], "Moon") == 5.0
