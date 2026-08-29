# Final accepted test suite for EvoEval_difficult/27
# 1 test function(s), mutation score computed over 6 mutant(s)

def check(candidate):
    assert candidate('Hello', 0) == 'hELLO'
    assert candidate('Hello', 5) == 'hELLO'
    assert candidate('hello', 0) == 'hELLO'
    assert candidate('a', 1) == 'a'
    assert candidate('abc', -1) == 'ABc'

    def helper(s, idx):
        if idx >= len(s):
            idx %= len(s)
        result = []
        for i, c in enumerate(s):
            if i == idx:
                if c.isupper():
                    result.append(c.lower())
                else:
                    result.append(c)
            elif c.isupper():
                result.append(c.lower())
            else:
                result.append(c.upper())
        return ''.join(result)
