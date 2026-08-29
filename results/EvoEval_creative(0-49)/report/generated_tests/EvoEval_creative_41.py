# Final accepted test suite for EvoEval_creative/41
# 4 test function(s), mutation score computed over 22 mutant(s)

def check(candidate):
    # test 1: requires moving down from the start (exposes COMPARISON_FLIP / NEGATE_CONDITION)
    arr1 = [[1, 99],
            [1, 0]]
    assert candidate(arr1) == 2

    # test 2: requires moving right to reach safe cell (exposes CONSTANT_CHANGE neighbor list bug)
    arr2 = [[1, 1, 0],
            [99, 99, 99]]
    assert candidate(arr2) == 2

    # test 3: relies on correct initialization of dp[0][0] (exposes INDEX_BOUNDARY mutations)
    arr3 = [[1, 2],
            [3, 0]]
    assert candidate(arr3) == 3

def check(candidate):
    # vertical 3x1: canonical minimal energy = 7 + 1 + 0 = 8
    assert candidate([[7], [1], [0]]) == 8
    # single-row 1x3 to expose INDEX_BOUNDARY (mutant uses arr[1])
    # canonical minimal energy = 5 + 4 + 0 = 9
    assert candidate([[5, 4, 0]]) == 9

def check(candidate):
    assert candidate([[1, 1], [100, 1], [0, 1]]) == 4
    assert candidate([[1, 100, 0], [1, 1, 1]]) == 4
    assert candidate([[5, 0]]) == 5

def check(candidate):
    n = 16
    grid = [[1] * n for _ in range(n)]
    grid[-1][-1] = 0
    assert candidate(grid) == 2 * n - 2
