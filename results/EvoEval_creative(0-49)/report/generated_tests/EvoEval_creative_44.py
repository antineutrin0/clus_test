# Final accepted test suite for EvoEval_creative/44
# 1 test function(s), mutation score computed over 10 mutant(s)

def check(candidate):
    assert candidate([5, 10, 15], 5, 20) == False
    assert candidate([10, 20, 30], 5, 10) == False
    assert candidate([1, 2, 3], 1, 5) == True
    assert candidate([5, 10, 15], 15, 15) == True
    assert candidate([], 0, 0) == True
    assert candidate([0], 1, 1) == True
