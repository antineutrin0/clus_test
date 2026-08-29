# Final accepted test suite for EvoEval_creative/5
# 1 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate([[1, 2], [3, 4], [5, 6, 7], [8, 9, 10]]) == [1, 2, 4, 3, 5, 6, 7, 10, 9, 8]
    assert candidate([]) == []

    def zigzag_traversal(matrix):
        result = []
        direction = 1
        for row in matrix:
            if direction == 1:
                result.extend(row)
