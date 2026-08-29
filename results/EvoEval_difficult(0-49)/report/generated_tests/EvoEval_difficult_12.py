# Final accepted test suite for EvoEval_difficult/12
# 1 test function(s), mutation score computed over 9 mutant(s)

def check(candidate):
    assert candidate([]) == (None, None)
    assert candidate(['apple', 'banana', 'cherry']) == ('apple', 0)
    assert candidate(['grape', 'blueberry', 'strawberry']) == ('ueberry', 1)
    assert candidate(["a"]) == ('a', 0)
    assert candidate(["a", "b"]) == ('a', 0)
