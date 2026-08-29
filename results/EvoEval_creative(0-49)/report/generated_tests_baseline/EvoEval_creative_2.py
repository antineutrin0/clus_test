# Baseline iterative tests for EvoEval_creative/2

def baseline_check_0(candidate):
    assert candidate(["Y"], 1) == ["z"]
    assert candidate(["z"], 1) == ["a"]
    assert candidate(["z"], "abc") == ["h"]
