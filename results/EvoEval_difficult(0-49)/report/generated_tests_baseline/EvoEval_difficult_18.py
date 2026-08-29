# Baseline iterative tests for EvoEval_difficult/18

def baseline_check_0(candidate):
    assert candidate("aaa", ["aa", "a", "z", "a"]) == {"aa": 2, "a": 6, "z": 0}

def baseline_check_1(candidate):
    assert candidate("abc", ["", "a"]) == "Substrings cannot be empty"
