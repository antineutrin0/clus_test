# Final accepted test suite for EvoEval_difficult/3
# 3 test function(s), mutation score computed over 20 mutant(s)

def check(candidate):
    assert candidate([('deposit', 100), ('withdrawal', 50)], 200) == 'All operations successful'
    assert candidate([('deposit', 100), ('withdrawal', 150)], 100) == 'Daily limit exceeded'
    assert candidate([('deposit', 100), ('withdrawal', 150)], 200) == 'Balance below zero'
    assert candidate([], 0) == 'All operations successful'
    assert candidate(["a"], 1) == 'All operations successful'
    assert candidate(["a", "b"], -1) == 'All operations successful'

def check(candidate):
    assert candidate([('deposit', 100), ('withdrawal', 100)], 200) == 'All operations successful'
    assert candidate([('deposit', 200), ('withdrawal', 60), ('withdrawal', 40)], 100) == 'All operations successful'
    assert candidate([('deposit', 200), ('withdrawal', 60), ('withdrawal', 50)], 100) == 'Daily limit exceeded'

def check(candidate):
    assert candidate([('withdrawal', 1)], 10) == 'Balance below zero'
