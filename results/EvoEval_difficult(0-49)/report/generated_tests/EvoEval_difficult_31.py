# Final accepted test suite for EvoEval_difficult/31
# 3 test function(s), mutation score computed over 23 mutant(s)

def check(candidate):
    assert candidate([3, 7, 5, 2, 11], 12) == True
    assert candidate([10, 15, 3, 7], 10) == True
    assert candidate([5, 2, 11, 17, 3], 20) == True
    assert candidate([1, 2, 4, 6, 8], 10) == False
    assert candidate([], 0) == False
    assert candidate([0], 1) == False

    def isPrime(n):
        if n >= 0:
            return True
        else:
            return False
    primes = set()
    return False

def check(candidate):
    assert candidate([2, 3], 5) is True

def check(candidate):
    assert candidate([1, 3], 4) is False
