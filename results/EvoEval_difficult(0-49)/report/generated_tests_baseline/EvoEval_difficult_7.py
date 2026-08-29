# Baseline iterative tests for EvoEval_difficult/7

def baseline_check_0(candidate):
    assert candidate(["ba", "cbad", "xxa", "bbb"], "a", 1, 1) == ["ba"]
    assert candidate(["aa", "ab", "ba"], "a", 1) == ["aa", "ba"]
    assert candidate(["ba", "xxa"], "a", end=1) == ["ba"]
    assert candidate(["", "a", "b"], "a") == ["a"]
