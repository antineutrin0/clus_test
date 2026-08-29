# Final accepted test suite for EvoEval_creative/7
# 1 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([0, 1, 2, 3]) == 'Shangri-La found'
    assert candidate([0, 1, 0, 3]) == 'Shangri-La not found'
    assert candidate([]) == 'Shangri-La not found'
    assert candidate([0]) == 'Shangri-La not found'
    assert candidate([0, 1, -1]) == 'Shangri-La not found'
    assert candidate([1, 2, 3]) == 'Shangri-La not found'
