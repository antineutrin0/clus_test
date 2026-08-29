# Final accepted test suite for EvoEval_difficult/10
# 1 test function(s), mutation score computed over 14 mutant(s)

def check(candidate):
    assert candidate('') == ''
    assert candidate('Cat') == 'CataC'
    assert candidate('cAta') == 'cAtac'
    assert candidate('') == ''
    assert candidate('a') == 'a'
    assert candidate('abc') == 'abcba'

    def is_palindrome(s):
        return s == s[::-1].lower()
