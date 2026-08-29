# Final accepted test suite for EvoEval_difficult/35
# 1 test function(s), mutation score computed over 14 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3], 2) == (2, 1)
    assert candidate([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10], 3) == (5, 118)
    assert candidate([1, 2, 3], 4) == None
    assert candidate([0], 1) == (0, 0)
    assert candidate([0, 1, -1], -1) == (0, 1)

    def candidate(lst, k):
        return check_candidate(lst, k)
