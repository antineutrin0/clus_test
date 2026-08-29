# Final accepted test suite for EvoEval_creative/19
# 1 test function(s), mutation score computed over 19 mutant(s)

def check(candidate):
    assert candidate([[0, 1, 0], [2, 0, 1], [1, 1, 0]]) == 2
    assert candidate([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0
    assert candidate([[1, 2, 1], [1, 1, 1], [2, 1, 1]]) == 6
    assert candidate([[1, 1, 1], [0, 0, 0], [2, 2, 2]]) == 3
    assert candidate([[2]]) == 0
    assert candidate([]) == 0
