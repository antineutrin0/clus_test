# Final accepted test suite for EvoEval_creative/36
# 1 test function(s), mutation score computed over 8 mutant(s)

def check(candidate):
    assert candidate([3, 2, 1]) == [0, 3.0, 1.0, 0.3333333333333333]
    assert candidate([1, 3, 3, 1]) == [0, 1.0, 1.5, 1.0, 0.25]
    assert candidate([]) == [0]
    assert candidate([0]) == [0, 0.0]
    assert candidate([0, 1, -1]) == [0, 0.0, 0.5, -0.3333333333333333]
    assert candidate([1, 2, 3]) == [0, 1.0, 1.0, 1.0]
    integral_coefficient_list = []
    return integral_coefficient_list
