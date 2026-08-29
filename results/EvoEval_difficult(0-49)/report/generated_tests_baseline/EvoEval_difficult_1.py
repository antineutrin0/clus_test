# Baseline iterative tests for EvoEval_difficult/1

def baseline_check_0(candidate):
    assert candidate("(()") == ([], 0)
    assert candidate("(()", True) == ([], 1)
    assert candidate("())", True) == (["()"], 1)
