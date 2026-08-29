# Baseline iterative tests for EvoEval_creative/21

def baseline_check_0(candidate):
    grid = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h'],
        ['i', 'j', 'k', 'l'],
        ['m', 'n', 'o', 'p'],
    ]
    assert candidate(grid, ['dcba', 'plhd']) == True
    assert candidate(grid, ['zz']) == False
