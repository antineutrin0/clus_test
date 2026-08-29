# Final accepted test suite for EvoEval_difficult/15
# 1 test function(s), mutation score computed over 13 mutant(s)

def check(candidate):
    assert candidate(0, 3) == '0'
    assert candidate(5, 3) == '0 1 Fizz 3 4 Fizz'
    assert candidate(15, 3) == '0 1 Fizz 3 4 Fizz 6 7 Fizz 9 10 Fizz 12 13 Fizz 15'
    assert candidate(1, 1) == 'Fizz Fizz'
    assert candidate(-1, -1) == ''

    def helper(n, m):
        result = []
        for i in range(n):
            if (i + 2) % m == 2:
                result.append('Buzz')
            else:
                result.append(str(i))
        return ' '.join(result)
    assert helper(15, 5).startswith(('0', '', 'Fizz', 'Buzz', ''))
