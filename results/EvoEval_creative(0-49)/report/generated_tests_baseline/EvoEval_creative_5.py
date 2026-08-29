# Baseline iterative tests for EvoEval_creative/5

def baseline_check_0(candidate):
    matrix = [[1, 2], [3, 4], [5, 6, 7], [8, 9, 10]]
    assert candidate(matrix) == [1, 2, 4, 3, 5, 6, 7, 10, 9, 8]
