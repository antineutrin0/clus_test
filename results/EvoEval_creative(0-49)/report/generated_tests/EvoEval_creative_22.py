# Final accepted test suite for EvoEval_creative/22
# 1 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(1, 300000) == 0.9999788605855618
    assert candidate(0, 0) == 0.0
    assert candidate(1, 1) == 3.3332628686185394e-06
    assert candidate(-1, -1) == 3.3332628686185394e-06
    assert candidate(2, 2) == 1.3333051474474157e-05
