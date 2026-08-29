# Baseline iterative tests for EvoEval_creative/24

def baseline_check_0(candidate):
    assert candidate("hello") == "#%((?"
    assert candidate("") == ""
