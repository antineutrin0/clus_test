# Final accepted test suite for EvoEval_difficult/43
# 2 test function(s), mutation score computed over 17 mutant(s)

def check(candidate):
    assert candidate([1, 3, 5, 0], 2) == False
    assert candidate([1, 3, -2, -1], 2) == True
    assert candidate([1, -1, 2, -2, 3, -3, 4, -4], 4) == True
    assert candidate([2, 4, -5, 3, 5, 7], 3) == True
    assert candidate([1], 1) == False
    assert candidate([], 0) == True

def check(candidate):
    assert candidate([1], 2) is False
