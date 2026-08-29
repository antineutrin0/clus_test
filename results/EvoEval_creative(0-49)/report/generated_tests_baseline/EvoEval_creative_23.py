# Baseline iterative tests for EvoEval_creative/23

def baseline_check_0(candidate):
    def first_primes(n):
        ps = []
        x = 2
        while len(ps) < n:
            is_p = True
            for p in ps:
                if p * p > x:
                    break
                if x % p == 0:
                    is_p = False
                    break
            if is_p:
                ps.append(x)
            x += 1
        return ps

    primes = first_primes(26)
    expected = f"{primes[0]}{primes[25]}Z!"
    assert candidate("azZ!") == expected
