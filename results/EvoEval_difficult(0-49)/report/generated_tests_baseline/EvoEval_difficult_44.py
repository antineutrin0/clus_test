# Baseline iterative tests for EvoEval_difficult/44

def baseline_check_0(candidate):
    assert candidate("A", 16, 16) == "A"
    assert candidate("1", 2, 2) == "1"
    assert candidate("G", 16, 10) == "invalid base"
    assert candidate("10", 10, 10) == "10"

def baseline_check_1(candidate):
    assert candidate("0", 10, 2) == "0"
    assert candidate("9", 10, 10) == "9"
