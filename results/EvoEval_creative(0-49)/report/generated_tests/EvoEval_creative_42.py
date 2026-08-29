# Final accepted test suite for EvoEval_creative/42
# 1 test function(s), mutation score computed over 20 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3, 4, 5]) == (True, 4)
    assert candidate([5]) == (True, -1)
    assert candidate([1, 3, 4]) == (False, None)
    assert candidate([]) == (False, None)
    assert candidate([0]) == (False, None)
    assert candidate([0, 1, -1]) == (False, None)
