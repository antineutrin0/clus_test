# Final accepted test suite for EvoEval_difficult/46
# 6 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(5, 1) == 4
    assert candidate(6, 2) == 4
    assert candidate(7, 3) == 4
    assert candidate(0, 0) == 0
    assert candidate(1, 1) == 0
    assert candidate(-1, -1) == 'Invalid input'

def check(candidate):
    assert candidate(2, 2) == 0

def check(candidate):
    assert candidate(3, 1) == 2

def check(candidate):
    result = candidate(2, 1)
    assert result == 0

def check(candidate):
    assert candidate(4, 1) == 2
    assert candidate(4, 4) == 0
    assert candidate(8, 1) == 28
    assert candidate(8, 2) == 14

def check(candidate):
    a = candidate(1, 0)
    b = candidate(1, -2)
    assert a == b
