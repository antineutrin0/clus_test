# Final accepted test suite for EvoEval_difficult/42
# 1 test function(s), mutation score computed over 10 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3], [2]) == [2, 2, 4]
    assert candidate([5, 3, 5, 2, 3, 3, 9, 0, 123], [5, 3, 123]) == [5, 3, 5, 3, 3, 3, 10, 1, 123]
    assert candidate([5, 3, 5, 2, 3, 3, 9, 0, 123], [5, 3, '123']) == 'Ignore list should only contain integers'
    assert candidate([], []) == []
    assert candidate([0], [0]) == [0]
    assert candidate([0, 1, -1], [0, 1, -1]) == [0, 1, -1]
