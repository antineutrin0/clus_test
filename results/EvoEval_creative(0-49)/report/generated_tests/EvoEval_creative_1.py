# Final accepted test suite for EvoEval_creative/1
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([3.0, 'Book', 7.0], 'National') == 37.5
    assert candidate([1.0, 2.0, 3.0, 4.0], 'Local') == 20.0
    assert candidate([5.5, 2.0, 3.0], 'International') == 62.5
    assert candidate([], "") == 0.0
    assert candidate(["a"], "a") == 5.0
    assert candidate(["a", "b"], "abc") == 10.0

def check(candidate):
    assert candidate([5.0], 'Local') == 5.0
