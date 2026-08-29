# Final accepted test suite for EvoEval_creative/45
# 1 test function(s), mutation score computed over 5 mutant(s)

def check(candidate):
    assert candidate([1,2,3,2,5,3,6,4,8,2,7]) == 10
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 8
    assert candidate([9, 8, 7, 6, 5, 4, 3, 2, 1]) == 8
    assert candidate([5, 5, 5, 5, 5]) == 4
    assert candidate([]) == -1
    assert candidate([0]) == 0
