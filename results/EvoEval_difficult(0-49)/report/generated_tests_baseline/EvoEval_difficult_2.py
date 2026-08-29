# Baseline iterative tests for EvoEval_difficult/2

def baseline_check_0(candidate):
    assert candidate(3.5, 2) == "0.50"
    assert candidate(5.0, 0) == "0"
