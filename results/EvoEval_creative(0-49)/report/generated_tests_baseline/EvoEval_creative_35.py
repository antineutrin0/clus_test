# Baseline iterative tests for EvoEval_creative/35

def baseline_check_0(candidate):
    assert candidate("myemail@domain.com") is True
    assert candidate("myemail.domain.com") is False
    assert candidate("@a.b") is False
    assert candidate("ab.@c") is False
    assert candidate("ab@.") is True
