# Final accepted test suite for EvoEval_creative/3
# 2 test function(s), mutation score computed over 23 mutant(s)

def check(candidate):
    assert candidate('123') == ['abc', 'aw', 'lc']
    assert candidate('111') == ['aaa', 'ak', 'ka']
    assert candidate("") == ['']

def check(candidate):
    assert candidate('36') == ['cf']
