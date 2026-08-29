# Final accepted test suite for EvoEval_creative/12
# 1 test function(s), mutation score computed over 4 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3, 2, 1, 2, 5, 1, 6]) == 23
    assert candidate([0]) == 0
    assert candidate([0, 1, -1]) == 0
    assert candidate([1, 2, 3]) == 6
