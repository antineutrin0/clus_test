# Final accepted test suite for EvoEval_difficult/13
# 1 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate([3,5]) == -1
    assert candidate([25, 15, 35]) == 5
    assert candidate([48, 60, 36]) == 12
    assert candidate([5, 10, 15, 20, 25]) == 5
    assert candidate([0]) == 0
