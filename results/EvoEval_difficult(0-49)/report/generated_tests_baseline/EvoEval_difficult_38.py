# Baseline iterative tests for EvoEval_difficult/38

def baseline_check_0(candidate):
    assert candidate("cdfb", 4) == "bcdf"
    assert candidate("b", 1) == "b"
