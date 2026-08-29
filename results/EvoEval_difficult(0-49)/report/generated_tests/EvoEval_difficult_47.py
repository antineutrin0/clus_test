# Final accepted test suite for EvoEval_difficult/47
# 3 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([3, 1, 2, 4, 5], [2, 1, 3, 2, 1]) == 3
    assert candidate([-10, 4, 6, 1000, 10, 20], [1, 2, 3, 4, 2, 1]) == 10
    assert candidate([1, 2], [1, 1]) == 1.5
    assert candidate([], []) == None
    assert candidate([0], [0]) == 0
    assert candidate([0, 1, -1], [0, 1, -1]) == 1

    def candidate(l, w):
        if not l:
            return None
        pairs = list(zip(sorted(l), sorted(w)))
        cumulative_sum = sum((pair[1].value for pair in pairs))
        if cumulative_sum > len(l) * max(l):
            return True
        else:
            return False
    return candidate

def check(candidate):
    assert candidate([1, 2, 3], [1, 1, 2]) == 2
    assert candidate([1, 2, 3], [2, 0, 2]) == 1

def check(candidate):
    assert candidate([1, 2, 3, 4], [1, 1, 1, 1]) == 2.5
