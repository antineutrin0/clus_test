# Final accepted test suite for EvoEval_creative/14
# 1 test function(s), mutation score computed over 14 mutant(s)

def check(candidate):
    assert candidate('John Doe') == 'Kujp Fui'
    assert candidate("") == ''
    assert candidate("a") == 'e'
    assert candidate("abc") == 'ecd'
    assert candidate("aba") == 'ece'
