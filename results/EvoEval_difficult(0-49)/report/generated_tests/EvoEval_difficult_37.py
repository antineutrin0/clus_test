# Final accepted test suite for EvoEval_difficult/37
# 1 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3], [3, 2, 1]) == [3, 2, 1]
    assert candidate([5, 6, 3, 4], [6, 4, 3]) == [3, 6, 5, 4]
    assert candidate([7, 8, 5, 6], [5, 7, 8]) == [5, 8, 7, 6]
    assert candidate([], []) == []
    assert candidate([0], [0]) == [0]
    assert candidate([0, 1, -1], [0, 1, -1]) == [0, 1, -1]
