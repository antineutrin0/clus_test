# Final accepted test suite for EvoEval_difficult/44
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('8', 10, 3) == '22'
    assert candidate('8', 10, 2) == '1000'
    assert candidate('7', 10, 2) == '111'
    assert candidate('1A', 16, 10) == '26'
    assert candidate('Z', 36, 16) == '23'
    assert candidate("", 0, 0) == 'invalid base'

def check(candidate):
    assert candidate('9', 10, 11) == '9'
    assert candidate('0', 10, 2) == '0'
    assert candidate('A', 11, 11) == 'A'
