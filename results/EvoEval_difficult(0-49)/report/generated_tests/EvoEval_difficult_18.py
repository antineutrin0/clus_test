# Final accepted test suite for EvoEval_difficult/18
# 2 test function(s), mutation score computed over 23 mutant(s)

def check(candidate):
    assert candidate('', ['a']) == {'a': 0}
    assert candidate('aaa', ['a', 'a']) == {'a': 6}
    assert candidate('aaaa', ['aa', 'a']) == {'aa': 3, 'a': 4}
    assert candidate('abcabc', ['a', 'b', 'c']) == {'a': 2, 'b': 2, 'c': 2}
    assert candidate("", []) == {}

def check(candidate):
    assert candidate('abc', ['']) == 'Substrings cannot be empty'
