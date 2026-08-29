# Baseline iterative tests for EvoEval_creative/9

def baseline_check_0(candidate):
    assert candidate("Hi! hi, HI? bye.") == {"hi": 3, "bye": 1}
