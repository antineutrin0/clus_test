# Final accepted test suite for EvoEval_difficult/24
# 3 test function(s), mutation score computed over 23 mutant(s)

def check(candidate):
    assert candidate(15, 1) == 5
    assert candidate(15, 2) == 3
    assert candidate(15, 3) == -1
    assert candidate(1, 1) == -1

def check(candidate):
    assert candidate(2, 1) == -1
    assert candidate(12, 1) == 3

def check(candidate):
    assert candidate(4, 1) == 2
    assert candidate(6, 2) == 2
