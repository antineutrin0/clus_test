# Baseline iterative tests for EvoEval_creative/13

def baseline_check_0(candidate):
    assert candidate(["z", "a", "a"]) == "z"
    assert candidate([]) is None
