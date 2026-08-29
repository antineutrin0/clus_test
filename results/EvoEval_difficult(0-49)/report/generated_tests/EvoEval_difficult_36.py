# Final accepted test suite for EvoEval_difficult/36
# 2 test function(s), mutation score computed over 17 mutant(s)

def check(candidate):
    assert candidate(50, [11, 13], 7) == 0
    assert candidate(78, [2, 5], 1) == 0
    assert candidate(79, [3, 7, 11], 3) == 1
    assert candidate(0, [], 0) == 0
    assert candidate(-1, [0, 1, -1], -1) == 0

def check(candidate):
    assert candidate(1, [1], 0) == 1
    assert candidate(1, [1], 1) == 0
    assert candidate(3, [3], 3) == 0
