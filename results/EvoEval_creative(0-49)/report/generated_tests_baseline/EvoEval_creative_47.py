# Baseline iterative tests for EvoEval_creative/47

def baseline_check_0(candidate):
    assert candidate({"sugar": 1}, {"sugar": 2}) is True
    assert candidate({"flour": 3}, {"flour": 3}) is True
    assert candidate({"eggs": 5}, {"eggs": 2}) is False
