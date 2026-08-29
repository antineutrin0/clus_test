# Final accepted test suite for EvoEval_difficult/19
# 2 test function(s), mutation score computed over 3 mutant(s)

def check(candidate):
    assert candidate('three one five', ['five', 'one', 'three']) == 'five one three'
    assert candidate('three one five three', ['five', 'one']) == 'five one three three'
    assert candidate("", []) == ''
    assert candidate("a", ["a"]) == 'a'
    assert candidate("abc", ["a", "b"]) == 'abc'
    assert candidate("aba", ["a", "b"]) == 'aba'

def check(candidate):
    assert candidate('two one three', ['one', 'two', 'three']) == 'one two three'
