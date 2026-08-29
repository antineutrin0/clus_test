# Baseline iterative tests for EvoEval_creative/7

def baseline_check_0(candidate):
    assert candidate([]) == 'Shangri-La not found'
    assert candidate([0, 1, 2, 3]) == 'Shangri-La found'
    assert candidate([3]) == 'Shangri-La not found'
