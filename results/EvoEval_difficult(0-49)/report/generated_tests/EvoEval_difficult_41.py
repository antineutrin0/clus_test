# Final accepted test suite for EvoEval_difficult/41
# 2 test function(s), mutation score computed over 16 mutant(s)

def check(candidate):
    assert candidate(0, []) == 0
    assert candidate(1, [0]) == 0
    assert candidate(-1, [0, 1, -1]) == 3
    assert candidate(2, [1, 2, 3]) == 9

    def candidate_wrapper(*args, **kwargs):
        return candidate(*args, **kwargs)

def check(candidate):
    speeds = [1, 0, 0, -1]
    res = candidate(4, speeds)
    s = [speed for speed in speeds if speed > 0]
    zero_speed = [speed for speed in speeds if speed == 0]
    expected = len(s) * len(s) + 2 * len(s) * len(zero_speed)
    assert res == expected
