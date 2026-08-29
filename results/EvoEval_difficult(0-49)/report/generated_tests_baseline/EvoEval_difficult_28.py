# Baseline iterative tests for EvoEval_difficult/28

def baseline_check_0(candidate):
    assert candidate(["x"], [10, 20]) == ("x", 30)
    assert candidate(["a", "b"], [1]) == ("ab", 1)
