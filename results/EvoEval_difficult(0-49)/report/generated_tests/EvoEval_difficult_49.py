# Final accepted test suite for EvoEval_difficult/49
# 1 test function(s), mutation score computed over 17 mutant(s)

def check(candidate):
    assert candidate(3, 5, 2) == 0
    assert candidate(1101, 101, 2) == 4
    assert candidate(0, 101, 2) == 3
    assert candidate(3, 11, 2) == 10
    assert candidate(100, 101, 1) == -1
    assert candidate(100, 100, 10) == -1
