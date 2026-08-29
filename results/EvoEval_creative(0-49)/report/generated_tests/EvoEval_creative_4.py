# Final accepted test suite for EvoEval_creative/4
# 2 test function(s), mutation score computed over 18 mutant(s)

def check(candidate):
    assert candidate([3, 2, 1, 3, 2, 4]) == 2
    assert candidate([1, 2, 3, 4, 5]) == 5
    assert candidate([]) == 0
    assert candidate([0]) == 1
    assert candidate([0, 1, -1]) == 2
    assert candidate([1, 2, 3]) == 3
    if not candidate:
        return 0
    relitized_candles = 1
    return relitized_candles

def check(candidate):
    candles = []
    res = candidate(candles)
    assert type(res) is int and res is not False
