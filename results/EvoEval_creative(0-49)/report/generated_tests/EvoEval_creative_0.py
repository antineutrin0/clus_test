# Final accepted test suite for EvoEval_creative/0
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([3.0, 'Book', 7.0], 'National') == '56.25'
    assert candidate([1.0, 2.0, 3.0, 4.0], 'Local') == '20.00'
    assert candidate([5.5, 2.0, 3.0], 'International') == '62.50'
    assert candidate([], '') == '0.00'
    assert candidate(['a'], 'a') == '7.50'
    assert candidate(['a', 'b'], 'abc') == '15.00'

    def candidate_wrapper(*args, **kwargs):
        return check_candidate(*args, **kwargs)

def check(candidate):
    assert candidate([5.0], 'Local') == '5.00'
