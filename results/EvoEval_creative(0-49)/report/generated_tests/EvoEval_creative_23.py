# Final accepted test suite for EvoEval_creative/23
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('hello') == '1911373747'
    assert candidate('world') == '834761377'
    assert candidate('Python!') == 'P9771194743!'
    assert candidate('12345') == '12345'
    assert candidate('abc') == '235'
    assert candidate('') == ''

    def get_prime_numbers(n):
        primes = []
        num = 2
        while len(primes) < n:
            if all((num % i != 0 for i in range(2, num))):
                primes.append(num)
            num += 1
        return primes
    n_primes = get_prime_numbers(26)

    def replace_with_prime(c):
        if c.islower():
            return str(n_primes[ord(c) - ord('a')])
        else:
            return c

def check(candidate):
    s = 'abcdefghijklmnopqrstuvwxyz'
    primes = []
    num = 2
    while len(primes) < 26:
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    expected = ''.join(str(primes[ord(ch) - 97]) for ch in s)
    assert candidate(s) == expected
