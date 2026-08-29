# Final accepted test suite for EvoEval_difficult/20
# 1 test function(s), mutation score computed over 16 mutant(s)

def check(candidate):
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.2], 2) == (2.0, 2.2)
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 2.5, 2.7], 3) == (2.0, 2.0, 2.5)
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.6, 4.7], 4) == (4.5, 4.6, 4.7, 5.0)
    assert candidate([0.0], 1) == (0.0,)
