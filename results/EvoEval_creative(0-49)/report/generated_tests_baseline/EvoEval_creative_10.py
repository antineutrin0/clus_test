# Baseline iterative tests for EvoEval_creative/10

def baseline_check_0(candidate):
    assert candidate("C| C|| C||| X|||| D|| B|||") == [1, 2, 2]
    assert candidate("D| D|| D||| D|||| D|||||") == [1, 2]
    assert candidate("B| B|| B||| B|||| B|||||") == [1, 2]
