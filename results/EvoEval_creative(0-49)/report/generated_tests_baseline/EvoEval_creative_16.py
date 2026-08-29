# Baseline iterative tests for EvoEval_creative/16

def baseline_check_0(candidate):
    r1 = candidate(1)
    assert r1 == 0
    assert type(r1) is int

    assert candidate(2) == 1
    assert candidate(3) == 2
    assert candidate(4) == 4

def baseline_check_1(candidate):
    r0 = candidate(0)
    assert r0 == 0
    assert type(r0) is int
