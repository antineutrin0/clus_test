# Final accepted test suite for EvoEval_creative/16
# 2 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate(1) == 0
    assert candidate(2) == 1
    assert candidate(3) == 2
    assert candidate(10) == 88
    assert candidate(20) == 10945
    assert candidate(30) == 1346268

def check(candidate):
    n = 1
    res = candidate(n)
    assert res == 0
    assert type(res) is int
