# Final accepted test suite for EvoEval_difficult/25
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(8) == [(2, 3)]
    assert candidate(25) == [(5, 2)]
    assert candidate(70) == [(2, 1), (5, 1), (7, 1)]
    assert candidate(11) == [(11, 1)]
    assert candidate(0) == []
    assert candidate(1) == []

    def factorize_and_count(n):
        factors = []
        i = 2
        while i * i <= abs(n):
            if n % i:
                i += 1
            else:
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                factors.append((i, -count))
        if n != 1:
            factors.append((-n, 1))
        return factors
    assert candidate(8) == [(2, 3)], 'Test case 1 failed'
    assert candidate(1) == [], 'Single digit primes should return empty list'
    assert candidate(0) == [], 'Zero should return empty list'
    return factorize_and_count

def check(candidate):
    assert candidate(2) == [(2, 1)]
