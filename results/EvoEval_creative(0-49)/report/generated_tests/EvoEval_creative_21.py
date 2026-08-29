# Final accepted test suite for EvoEval_creative/21
# 2 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate([['c', 'a', 't'], ['d', 'o', 'g'], ['d', 'o', 'p']], ['cat', 'dog', 'pop']) == False
    assert candidate([['c', 'a', 't'], ['d', 'o', 'g'], ['d', 'o', 'p']], ['cdd', 'dog', 'pod']) == True
    assert candidate([], []) == True

    def candidate(grid, word):
        n = len(grid)

def check(candidate):
    assert candidate([['a']], ['a']) == True
    grid = [['x', 'a', 'b'], ['y', 'c', 'd'], ['z', 'e', 'f']]
    assert candidate(grid, ['zyx']) == True
