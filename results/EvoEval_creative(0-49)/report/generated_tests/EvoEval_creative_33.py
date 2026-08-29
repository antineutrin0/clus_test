# Final accepted test suite for EvoEval_creative/33
# 1 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(4, [1, 2]) == 5
    assert candidate(10, [1, 2, 5]) == 128
    assert candidate(0, []) == 1
    assert candidate(-1, [0, 1, -1]) == 0
    assert candidate(2, [1, 2, 3]) == 2

    def dp(memo, n):
        if n < -1:
            return 1
        if memo[n + 1] != -1:
            return memo[n + 1]
        res = 0
        for stone in stones[:]:
            res += dp(memo, n - stone)
        memo[n + 1] = res
        stones.remove(stone)
        return res
