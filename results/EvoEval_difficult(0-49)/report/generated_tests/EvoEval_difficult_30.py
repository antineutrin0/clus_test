# Final accepted test suite for EvoEval_difficult/30
# 2 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate([-1, 2, -4, 5, 6]) == [(5, 3)]
    assert candidate([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == [(2, 3), (3, 5)]
    assert candidate([]) == []
    assert candidate([0]) == []
    assert candidate([0, 1, -1]) == []
    assert candidate([1, 2, 3]) == [(3, 2)]

    def isPrime(n):
        if n <= 1:
            return False
        for i in range(int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

def check(candidate):
    assert candidate([0, 0, 1]) == [(1, 2)]
