# Baseline iterative tests for EvoEval_creative/3

def baseline_check_0(candidate):
    assert candidate("") == [""]
    assert candidate("1") == ["a"]
    assert candidate("12") == ["ab", "l"]
    assert candidate("25") == ["be", "y"]
    assert candidate("30") == []
    assert candidate("121") == ["aba", "au", "la"]
