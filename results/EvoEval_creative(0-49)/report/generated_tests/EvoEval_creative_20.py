# Final accepted test suite for EvoEval_creative/20
# 1 test function(s), mutation score computed over 22 mutant(s)

def check(candidate):
    assert candidate(3) == ['1', '1,2', '1,2,3', '3,2,1', '2,1', '1']
    assert candidate(0) == []
    assert candidate(1) == ['1', '1']
    assert candidate(-1) == []
    assert candidate(2) == ['1', '1,2', '2,1', '1']
