# Final accepted test suite for EvoEval_creative/28
# 1 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(12000, 500) == 225.0
    assert candidate(55000, 2500) == 7750.0
    assert candidate(120000, 5000) == 23700.0
    assert candidate(0.0, 0) == 0
    assert candidate(1.0, 1) == 0
