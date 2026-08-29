# Final accepted test suite for EvoEval_difficult/48
# 1 test function(s), mutation score computed over 6 mutant(s)

def check(candidate):
    assert candidate('') == True
    assert candidate('Able , was I saw Elba') == True
    assert candidate('A man, a plan, a canal, Panama') == True
    assert candidate('This is not a palindrome') == False
    assert candidate('') == True
    assert candidate('a') == True

    def helper(s):
        s = re.sub('\\W+', '', s.lower())
        return s == s[::-1]
