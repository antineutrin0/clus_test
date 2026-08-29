# Final accepted test suite for EvoEval_difficult/9
# 1 test function(s), mutation score computed over 11 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3, 2, 3, 4, 2]) == [(1, 1), (2, 1), (3, 1), (3, 1), (3, 1), (4, 1), (4, 1)]
    assert candidate([]) == []
    assert candidate([0]) == [(0, 0)]
    assert candidate([0, 1, -1]) == [(0, 0), (1, 0), (1, -1)]
    assert candidate([1, 2, 3]) == [(1, 1), (2, 1), (3, 1)]
    result = []
    return result
