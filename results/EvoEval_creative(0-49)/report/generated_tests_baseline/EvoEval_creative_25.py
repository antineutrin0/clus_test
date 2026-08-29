# Baseline iterative tests for EvoEval_creative/25

def baseline_check_0(candidate):
    result = candidate("alpha key. beta key.", "key")
    assert result == (2, "alpha key")
