# Baseline iterative tests for EvoEval_difficult/11

def baseline_check_0(candidate):
    assert candidate("01001", "11100", "11100", (1, 3)) == "10010"
