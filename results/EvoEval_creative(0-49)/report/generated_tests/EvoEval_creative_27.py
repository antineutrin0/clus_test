# Final accepted test suite for EvoEval_creative/27
# 1 test function(s), mutation score computed over 6 mutant(s)

def check(candidate):
    assert candidate('HelloWorld', 3) == ['Hel', 'loW', 'orl', 'd']
    assert candidate('CodingIsFun', 5) == ['Codin', 'gIsFu', 'n']
    assert candidate('a', 1) == ['a']
    assert candidate('abc', -1) == []
    assert candidate('aba', 2) == ['ab', 'a']

    def helper():
        try:
            result = candidate('')
            assert result == [], 'Empty string case failed'
        except Exception as e:
            print(f'Exception caught during empty string test: {e}')
