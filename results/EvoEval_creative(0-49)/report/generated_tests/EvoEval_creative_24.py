# Final accepted test suite for EvoEval_creative/24
# 1 test function(s), mutation score computed over 2 mutant(s)

def check(candidate):
    assert candidate("hello") == '#%((?'
    assert candidate("alien") == '+($%='
    assert candidate("") == ''
    assert candidate("a") == '+'
    assert candidate("abc") == '+-*'
    assert candidate("aba") == '+-+'
