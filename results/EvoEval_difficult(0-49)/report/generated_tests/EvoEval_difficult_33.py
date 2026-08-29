# Final accepted test suite for EvoEval_difficult/33
# 1 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3], [4, 5, 6]) == [4, 2, 3, 1, 5, 6]
    assert candidate([5, 6, 3, 4, 8, 9, 2], []) == [5, 6, 3, 4, 8, 9, 2]
    assert candidate([], [1, 2, 3]) == [1, 2, 3]
    assert candidate([], []) == []
    assert candidate([0], [0]) == [0, 0]
    assert candidate([0, 1, -1], [0, 1, -1]) == [0, 1, -1, 0, 1, -1]
