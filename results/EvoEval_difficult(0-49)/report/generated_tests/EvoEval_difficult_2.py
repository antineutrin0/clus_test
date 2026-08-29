# Final accepted test suite for EvoEval_difficult/2
# 1 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate(3.5, 2) == '0.50'
    assert candidate(4.12345, 3) == '0.123'
    assert candidate(0.0, 0) == '0'
    assert candidate(1.0, 1) == '0.0'
    assert candidate(0.5, 2) == '0.50'
