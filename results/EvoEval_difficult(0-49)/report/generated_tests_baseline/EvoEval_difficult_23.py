# Baseline iterative tests for EvoEval_difficult/23

def baseline_check_0(candidate):
    assert candidate("a b") == 3
    assert candidate("a b1", True, False) == 3
