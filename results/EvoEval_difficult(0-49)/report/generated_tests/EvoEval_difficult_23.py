# Final accepted test suite for EvoEval_difficult/23
# 2 test function(s), mutation score computed over 7 mutant(s)

def check(candidate):
    assert candidate('', True, True) == 0
    assert candidate('abc') == 3
    assert candidate('abc def', True) == 6
    assert candidate('abc123', False, True) == 3
    assert candidate('abc def123', True, True) == 6
    assert candidate("", False, False) == 0

def check(candidate):
    assert candidate('abc def') == 7
