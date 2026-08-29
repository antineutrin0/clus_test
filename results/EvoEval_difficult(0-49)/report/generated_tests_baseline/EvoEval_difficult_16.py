# Baseline iterative tests for EvoEval_difficult/16

def baseline_check_0(candidate):
    assert candidate("aba", 3) == {"aba": 2}
    assert candidate("abcd", 2) == {"ab": 2, "bc": 2, "cd": 2}
    assert candidate("abc", 5) == {}
