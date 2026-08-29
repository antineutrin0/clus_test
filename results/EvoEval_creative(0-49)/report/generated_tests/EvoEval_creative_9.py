# Final accepted test suite for EvoEval_creative/9
# 1 test function(s), mutation score computed over 9 mutant(s)

def check(candidate):
    assert candidate('Hello, World! Hello Again.') == {'hello': 2, 'world': 1, 'again': 1}
    assert candidate('This is a test. This is only a test.') == {'this': 2, 'is': 2, 'a': 2, 'test': 2, 'only': 1}
    assert candidate("") == {}
    assert candidate("a") == {'a': 1}
    assert candidate("abc") == {'abc': 1}
    assert candidate("aba") == {'aba': 1}
